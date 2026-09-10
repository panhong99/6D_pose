"""Automatic detect/register/track state machine; camera and model independent."""
import time


class RecoveryTracker:
    def __init__(self, detector, tracker, prompt="rubik's cube", retry_interval=0.5,
                 max_frame_gap=1.0, validation_interval=0.0, loss_patience=5):
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

    def reset(self, restart_search=True):
        self.tracker.reset()
        self.tracking = False
        self.last_stamp = None
        self.next_attempt = 0.0
        self.next_validation = 0.0
        self.detector_misses = 0
        self.consecutive_failures = 0
        if restart_search:
            self.search_attempt = 0
            self.search_started = time.perf_counter()

    def _lost(self, outcome, frame_id, reason):
        print(f"[LOST] frame={frame_id} reason={reason}", flush=True)
        self.reset()
        outcome.update(pose=None, error=reason, status=f'LOST: {reason}; automatic recovery')
        return outcome

    def process(self, frame):
        outcome = dict(pose=None, detection=None, error=None, registered=False,
                       status='Searching for cube', suspect=False)
        if self.tracking and self.last_stamp is not None:
            gap = (frame.timestamp_ns - self.last_stamp) / 1e9
            if gap <= 0 or gap > self.max_frame_gap:
                print(f"[LOST] reason=frame_gap gap_s={gap:.3f}", flush=True)
                self.reset()
        if self.tracking:
            frame_id = getattr(frame, 'identifier', '?')
            try:
                outcome['pose'] = self.tracker.estimate(frame)  # fatal on failure (NaN/no prior)
            except ValueError as exc:
                return self._lost(outcome, frame_id, str(exc))

            # Soft signals below: pose exists, but confidence/geometry/DINO disagree.
            # loss_patience consecutive soft failures force a re-detect; a double
            # DINO miss (nothing found at all) is deliberate enough to skip that wait.
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
                    if self.detector_misses >= 2:
                        hard_reason = 'Lost: DINO missed target twice'
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
        print(f"[SEARCH_BEGIN] frame={getattr(frame, 'identifier', '?')} "
              f"attempt={self.search_attempt} retry={self.search_attempt-1}", flush=True)
        try:
            detection = self.detector.infer(frame.rgb, self.prompt)
            outcome['detection'] = detection
            if not detection['success']:
                outcome['status'] = 'SEARCHING: no valid cube mask; retrying'
                return outcome
            estimate_start = time.perf_counter()
            try:
                outcome['pose'] = self.tracker.estimate(frame, detection['mask'])
            finally:
                estimate_ms = (time.perf_counter() - estimate_start) * 1000
            if self.validation_interval > 0:
                self.tracker.validate_geometry(frame, outcome['pose'], detection['bbox'])
            self.tracking = True
            # Registration can take longer than max_frame_gap. Do not interpret
            # that expected startup delay as an immediate tracking loss.
            self.last_stamp = None
            self.next_validation = time.perf_counter() + self.validation_interval
            self.detector_misses = 0
            outcome.update(registered=True, status='REGISTERED -> TRACKING')
        except ValueError as exc:
            if detection is None:
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
                  f"attempt={self.search_attempt} retry={self.search_attempt-1} "
                  f"dino_ms={timings.get('dino_ms')} sam_ms={timings.get('sam_ms')} "
                  f"detect_total_ms={timings.get('latency_ms')} estimate_total_ms={estimate_ms} "
                  f"cycle_ms={(ended-cycle_start)*1000:.1f} "
                  f"search_elapsed_ms={(ended-self.search_started)*1000:.1f} "
                  f"status={outcome['status']}", flush=True)
            self.next_attempt = time.perf_counter() + self.retry_interval
        return outcome
