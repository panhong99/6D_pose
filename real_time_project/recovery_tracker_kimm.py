"""Automatic detect/register/track state machine; camera and model independent."""
import time

import numpy as np


class RecoveryTracker:
    def __init__(self, detector, tracker, prompt="rubik's cube", retry_interval=0.5,
                 max_frame_gap=1.0, validation_interval=0.0, loss_patience=5,
                 roi_margin=1.5, roi_patience=3):
        self.detector, self.tracker = detector, tracker
        self.prompt = prompt
        self.retry_interval, self.max_frame_gap = retry_interval, max_frame_gap
        self.tracking = False
        self.last_stamp = None
        self.next_attempt = 0.0
        self.validation_interval = validation_interval
        self.next_validation = 0.0
        self.detector_misses = 0
        self.search_attempt = 0
        self.search_started = None
        # Consecutive soft failures tolerated before forcing a full re-detect.
        self.loss_patience = loss_patience
        self.consecutive_failures = 0
        self.auto_recovery = False
        self.waiting_for_manual_search = False
        # Last known image-space bbox (from a live track or a past detection),
        # used to crop the detector search instead of scanning the full frame.
        self.roi_margin = roi_margin
        self.roi_patience = roi_patience
        self._last_bbox = None
        self._roi_misses = 0

    def set_recovery_mode(self, auto_recovery=False):
        self.auto_recovery = bool(auto_recovery)

    def request_search(self):
        """Clear a manual-recovery pause and start searching on the next frame."""
        self.reset()
        self.waiting_for_manual_search = False

    def reset(self, restart_search=True):
        # Keep the pre-loss orientation so the next search can try a cheap relock
        # before falling back to a full re-registration; _last_bbox is kept too
        # (not cleared here) since it stays a useful ROI hint across the loss.
        self.tracker.reset(keep_prior=True)
        self.tracking = False
        self.last_stamp = None
        self.next_attempt = 0.0
        self.next_validation = 0.0
        self.detector_misses = 0
        self.consecutive_failures = 0
        if restart_search:
            self.search_attempt = 0
            self.search_started = time.perf_counter()
            self._roi_misses = 0

    def _lost(self, outcome, frame_id, reason):
        print(f"[LOST] frame={frame_id} reason={reason}", flush=True)
        self.reset()
        if self.auto_recovery:
            status = f'LOST: {reason}; automatic recovery'
        else:
            self.waiting_for_manual_search = True
            status = f'LOST: {reason}; press S to search'
        outcome.update(pose=None, error=reason, status=status)
        return outcome

    def process(self, frame):
        outcome = dict(pose=None, detection=None, error=None, registered=False,
                       status='Searching for cube', suspect=False)
        if self.waiting_for_manual_search:
            outcome['status'] = 'LOST: press S to search'
            return outcome
        if self.tracking and self.last_stamp is not None:
            gap = (frame.timestamp_ns - self.last_stamp) / 1e9
            if self.auto_recovery and (gap <= 0 or gap > self.max_frame_gap):
                print(f"[LOST] reason=frame_gap gap_s={gap:.3f}", flush=True)
                self.reset()
        if self.tracking:
            frame_id = getattr(frame, 'identifier', '?')
            try:
                outcome['pose'] = self.tracker.estimate(frame)  # fatal on failure (NaN/no prior)
            except ValueError as exc:
                return self._lost(outcome, frame_id, str(exc))
            self._last_bbox = self.tracker.project_bbox(frame.K, outcome['pose']) or self._last_bbox

            # Manual mode deliberately does not run the automatic loss
            # watchdog. The operator decides when the pose should be
            # reacquired by pressing S.
            if not self.auto_recovery:
                self.last_stamp = frame.timestamp_ns
                outcome['status'] = 'TRACKING (manual recovery)'
                return outcome

            # Soft signals below: pose exists, but confidence/geometry/DINO disagree.
            # Treat detector misses like other soft failures; brief blur should not
            # immediately reset a valid FoundationPose track.
            reason = None if self.tracker.last_score_ok else 'score drift'
            hard_reason = None
            if self.validation_interval > 0:
                try:
                    self.tracker.validate_geometry(frame, outcome['pose'])
                except ValueError as exc:
                    reason = reason or str(exc)
                if time.perf_counter() >= self.next_validation:
                    box = self.detector.detect_bbox(frame.rgb, self.prompt)['bbox']
                    self.next_validation = time.perf_counter() + self.validation_interval
                    self.detector_misses = self.detector_misses + 1 if box is None else 0
                    if self.detector_misses >= self.loss_patience:
                        hard_reason = f'Lost: DINO missed target {self.loss_patience} times'
                    elif box is not None:
                        try:
                            self.tracker.validate_geometry(frame, outcome['pose'], box)
                        except ValueError as exc:
                            reason = reason or str(exc)

            self.last_stamp = frame.timestamp_ns
            if hard_reason:
                return self._lost(outcome, frame_id, hard_reason)
            if reason is None:
                self.consecutive_failures = 0
                outcome['status'] = 'TRACKING'
                return outcome
            self.consecutive_failures += 1
            print(f"[SUSPECT] frame={frame_id} reason={reason} "
                 f"streak={self.consecutive_failures}/{self.loss_patience}", flush=True)
            if self.consecutive_failures >= self.loss_patience:
                return self._lost(outcome, frame_id, reason)
            outcome['suspect'] = True
            outcome['status'] = f'SUSPECT ({self.consecutive_failures}/{self.loss_patience}): {reason}'
            return outcome

        if time.perf_counter() < self.next_attempt:
            outcome['status'] = 'SEARCHING: waiting to retry'
            return outcome
        # Use one RGB-D frame for detection, segmentation and pose registration.
        # Runtime/model failures propagate; ordinary misses remain retryable.
        cycle_start = time.perf_counter()
        if self.search_started is None:
            self.search_started = cycle_start
        self.search_attempt += 1
        detection = None
        estimate_ms = None
        source = None
        print(f"[SEARCH_BEGIN] frame={getattr(frame, 'identifier', '?')} "
              f"attempt={self.search_attempt} retry={self.search_attempt-1}", flush=True)
        try:
            # Tier 0: Cutie may still have a rough lock through the loss -- relock
            # straight from it and skip the detector entirely. A ValueError here
            # (e.g. score check) just means Tier 0 didn't pan out, not a fatal
            # detect error, so it's swallowed rather than left for the except
            # block below to potentially misclassify as fatal.
            try:
                box = self.tracker.cutie_probe(frame)
                pose = self.tracker.relock_from_bbox(frame, box) if box is not None else None
            except ValueError:
                pose = None
            if pose is not None:
                source = 'cutie'
            else:
                # Tier 1/2: crop to a ROI around the last known location so the
                # detector searches a small region instead of the full frame;
                # escalate to full-frame after roi_patience consecutive misses.
                roi = self._recovery_roi(frame)
                crop, offset = self._crop_rgb(frame.rgb, roi)
                detection = self.detector.infer(crop, self.prompt)
                if roi is not None:
                    detection = self._remap_detection(detection, offset, frame.rgb.shape[:2])
                outcome['detection'] = detection
                if not detection['success']:
                    if roi is not None:
                        self._roi_misses += 1
                    outcome['status'] = 'SEARCHING: no valid cube mask; retrying'
                    return outcome
                # tracker.estimate() itself tries a prior-seeded relock before
                # falling back to register()'s full multi-hypothesis search.
                estimate_start = time.perf_counter()
                try:
                    pose = self.tracker.estimate(frame, detection['mask'])
                finally:
                    estimate_ms = (time.perf_counter() - estimate_start) * 1000
                source = 'detector'
            outcome['pose'] = pose
            if self.validation_interval > 0 and detection is not None:
                self.tracker.validate_geometry(frame, pose, detection.get('bbox'))
            self.tracking = True
            # Registration can take longer than max_frame_gap. Do not interpret
            # that expected startup delay as an immediate tracking loss.
            self.last_stamp = None
            self.next_validation = time.perf_counter() + self.validation_interval
            self.detector_misses = 0
            self._last_bbox = self.tracker.project_bbox(frame.K, pose) or self._last_bbox
            self._roi_misses = 0
            outcome.update(registered=True, status=f'REGISTERED -> TRACKING ({source})')
        except ValueError as exc:
            if detection is None and source != 'cutie':
                outcome['status'] = f'DETECT error: {exc}'
                raise
            self.reset(restart_search=False)
            outcome.update(pose=None, error=str(exc), status=f'REGISTER failed: {exc}; retrying')
        except Exception as exc:
            outcome['status'] = f'ERROR: {exc}'
            raise
        finally:
            ended = time.perf_counter()
            timings = detection or {}
            print(f"[SEARCH_END] frame={getattr(frame, 'identifier', '?')} "
                  f"attempt={self.search_attempt} retry={self.search_attempt-1} source={source} "
                  f"dino_ms={timings.get('dino_ms')} sam_ms={timings.get('sam_ms')} "
                  f"detect_total_ms={timings.get('latency_ms')} estimate_total_ms={estimate_ms} "
                  f"cycle_ms={(ended-cycle_start)*1000:.1f} "
                  f"search_elapsed_ms={(ended-self.search_started)*1000:.1f} "
                  f"status={outcome['status']}", flush=True)
            self.next_attempt = time.perf_counter() + self.retry_interval
        return outcome

    def _recovery_roi(self, frame):
        if self._last_bbox is None or self._roi_misses >= self.roi_patience:
            return None
        x1, y1, x2, y2 = self._last_bbox
        w, h = x2 - x1, y2 - y1
        mx, my = max(w, 30) * self.roi_margin, max(h, 30) * self.roi_margin
        H, W = frame.rgb.shape[:2]
        cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        cx2, cy2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
        if cx2 - cx1 < 40 or cy2 - cy1 < 40:
            return None
        return cx1, cy1, cx2, cy2

    @staticmethod
    def _crop_rgb(rgb, roi):
        if roi is None:
            return rgb, (0, 0)
        x1, y1, x2, y2 = roi
        return np.ascontiguousarray(rgb[y1:y2, x1:x2]), (x1, y1)

    @staticmethod
    def _remap_detection(detection, offset, full_shape):
        ox, oy = offset
        if ox == 0 and oy == 0:
            return detection
        bbox = detection.get('bbox')
        if bbox is not None:
            detection['bbox'] = [bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy]
        mask = detection.get('mask')
        if mask is not None:
            full_mask = np.zeros(full_shape, dtype=mask.dtype)
            full_mask[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
            detection['mask'] = full_mask
        return detection
