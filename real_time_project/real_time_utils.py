"""Real-time tracking building blocks (YOLOE -> FoundationPose + Cutie/Kalman).

  D455Source, RGBDFrame, rectification_maps  aligned/rectified RGB-D input in meters
  PoseTracker      CAD loading, FoundationPose register/track, Cutie + Kalman, score/geometry checks
  RecoveryTracker  detect -> register -> track -> lost state machine (auto/manual recovery)
"""
from dataclasses import dataclass
import itertools
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import trimesh
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------- D455 input

def rectification_maps(intr, K):
    if intr.model == rs.distortion.none or not np.any(np.asarray(intr.coeffs) != 0):
        return None, None
    if intr.model == rs.distortion.brown_conrady:
        return cv2.initUndistortRectifyMap(
            K, np.asarray(intr.coeffs), None, K,
            (intr.width, intr.height), cv2.CV_32FC1)
    if intr.model not in (rs.distortion.inverse_brown_conrady,
                          rs.distortion.modified_brown_conrady):
        raise ValueError(f'Unsupported color distortion: {intr.model}')
    # A rectified output pixel defines an ideal ray. Ask the SDK where that ray
    # lands in the original image. Do not reinterpret inverse coefficients as
    # OpenCV Brown coefficients. This lookup is computed only once at startup.
    pixels = np.empty((intr.height, intr.width, 2), dtype=np.float32)
    for v in range(intr.height):
        y = float((v - K[1, 2]) / K[1, 1])
        for u in range(intr.width):
            x = float((u - K[0, 2]) / K[0, 0])
            pixels[v, u] = rs.rs2_project_point_to_pixel(intr, [x, y, 1.0])
    if not np.isfinite(pixels).all():
        raise ValueError('Camera calibration produced non-finite rectification maps')
    return pixels[..., 0].copy(), pixels[..., 1].copy()


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    timestamp_ns: int
    identifier: str


class D455Source:
    def __init__(self, width=640, height=480, fps=30, serial='', max_depth=3.0):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        try:
            device = profile.get_device()
            print('Camera:', device.get_info(rs.camera_info.name), flush=True)
            self.scale = device.first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            self.max_depth = max_depth
            self.map1 = self.map2 = None
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                              dtype=np.float32)
            # FoundationPose uses a pinhole K. Apply the same pixel mapping to RGB
            # and aligned depth so distortion does not silently violate that contract.
            self.map1, self.map2 = rectification_maps(intr, self.K)
        except Exception:
            self.pipeline.stop()
            raise

    def read(self):
        # Drain any frames queued while the caller was busy (e.g. a slow recovery
        # cycle): always process the newest one, not a growing backlog.
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        while True:
            newer = self.pipeline.poll_for_frames()
            if not newer:
                break
            frames = newer
        frames = self.align.process(frames)
        color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth_frame:
            raise RuntimeError('Missing synchronized RGB-D frame')
        # Device timestamps may be relative to camera startup, not Unix epoch.
        # This v1 deliberately publishes host receipt time, not a claimed capture time.
        timestamp_ns = time.time_ns()
        rgb = np.asanyarray(color.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.scale
        if self.map1 is not None:
            rgb = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self.map1, self.map2, cv2.INTER_NEAREST)
        depth[~np.isfinite(depth) | (depth < 0.001) | (depth > self.max_depth)] = 0
        return RGBDFrame(np.ascontiguousarray(rgb), depth, self.K.copy(),
                         timestamp_ns, str(color.get_frame_number()))

    def close(self):
        self.pipeline.stop()


# ---------------------------------------------------------------- pose tracking


class PoseTracker:
    def __init__(self, mesh_file, mesh_scale, debug_dir, register_iterations=5,
                 track_iterations=2, drift_score_ratio=0.3, detector_bbox_margin=0.5,
                 relock_iterations=6, score_interval=1):
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr
        mesh = trimesh.load(str(mesh_file), force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError('CAD must contain a triangle mesh')
        mesh.apply_scale(mesh_scale)
        if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
            raise ValueError('CAD must have finite vertices and nonzero 3D extent')
        print('CAD extent (meters):', mesh.extents, flush=True)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2])
        if debug_dir is not None:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
        self.est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(), debug=0,
            debug_dir=str(debug_dir) if debug_dir is not None else None)
        self.register_iterations = register_iterations
        self.track_iterations = track_iterations
        self.relock_iterations = relock_iterations
        # The drift check needs one extra scorer forward pass; run it every Nth tracking
        # frame (1 = every frame). Skipped frames keep the previous verdict.
        self.score_interval = max(1, int(score_interval))
        self._frames_since_score = 0
        # Orientation kept from the pose right before a loss, so a recent loss can be
        # re-locked from a single seeded hypothesis instead of register()'s full
        # multi-hypothesis orientation search (~40x more refiner/scorer work).
        self._prior_pose = None
        self.drift_score_ratio = drift_score_ratio
        # Detector boxes can move substantially under blur/rotation.  Keep this
        # guard loose; FoundationPose remains the primary pose signal.
        self.detector_bbox_margin = detector_bbox_margin
        self.two_d_tracker = None
        self.kf = None
        self.kf_mean = self.kf_cov = None
        try:
            import sys
            plus_src = str(Path(__file__).resolve().parents[1] / 'FoundationPose-plus-plus' / 'src')
            if plus_src not in sys.path:
                sys.path.insert(0, plus_src)
            from VOT import Cutie
            from utils.kalman_filter_6d import KalmanFilter6D
            self.two_d_tracker = Cutie()
            self.kf = KalmanFilter6D(0.05)
            print('2D tracker + Kalman enabled', flush=True)
        except Exception as exc:
            print(f'2D tracker disabled: {exc}', flush=True)
        self.score_ema_alpha = 0.2
        self.score_ema = None
        self.last_score = None
        self.last_score_ok = True
        self.registration_score = None

    def reset(self, keep_prior=False):
        if keep_prior and self.est.pose_last is not None:
            self._prior_pose = self.est.pose_last.detach().clone()
        else:
            self._prior_pose = None
        self.est.pose_last = None
        self.score_ema = None
        self.last_score = None
        self.last_score_ok = True
        self.registration_score = None

    def _score(self, frame):
        scores, _ = self.est.scorer.predict(
            mesh=self.est.mesh, rgb=frame.rgb, depth=frame.depth, K=frame.K,
            ob_in_cams=self.est.pose_last.reshape(1, 4, 4).data.cpu().numpy(),
            normal_map=None, mesh_tensors=self.est.mesh_tensors, glctx=self.est.glctx,
            mesh_diameter=self.est.diameter, get_vis=False)
        return float(scores[0])

    def estimate(self, frame, mask=None):
        """Raises only on a fatal failure. A low score sets last_score_ok=False
        instead of raising -- the caller decides how many bad frames mean real loss."""
        if mask is not None:
            return self._register_or_relock(frame, mask)
        return self._track(frame)

    def _register_or_relock(self, frame, mask):
        # relock_iterations == 0 disables the relock shortcut: always full register().
        prior_pose = self._prior_pose if self.relock_iterations > 0 else None
        self.reset()
        register_ms = relock_ms = None
        used_relock = False
        started = time.perf_counter()
        try:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != frame.depth.shape or np.count_nonzero(mask & (frame.depth > 0)) < 100:
                raise ValueError('Mask needs at least 100 valid depth pixels')
            pose = None
            if prior_pose is not None:
                relock_start = time.perf_counter()
                pose = self._relock(frame, mask, prior_pose)
                relock_ms = (time.perf_counter() - relock_start) * 1000
                used_relock = pose is not None
            if pose is None:
                register_start = time.perf_counter()
                try:
                    pose = self.est.register(K=frame.K, rgb=frame.rgb, depth=frame.depth,
                                             ob_mask=mask, iteration=self.register_iterations)
                finally:
                    # register returns a CPU pose, synchronizing its GPU work.
                    register_ms = (time.perf_counter() - register_start) * 1000
            # Validate the pose before spending a scorer forward pass on it: a NaN/None
            # pose_last would otherwise crash inside _score() instead of raising cleanly.
            if self.est.pose_last is None or not np.isfinite(pose).all():
                raise ValueError('Pose registration/tracking failed')
        except Exception:
            self.reset()
            raise
        finally:
            print(f"[REGISTER] frame={getattr(frame, 'identifier', '?')} "
                  f"relock_ms={relock_ms} used_relock={used_relock} "
                  f"register_ms={register_ms} "
                  f"estimate_total_ms={(time.perf_counter()-started)*1000:.1f}", flush=True)
        return self._finalize_lock(frame, mask, pose)

    def _track(self, frame):
        if self.est.pose_last is None:
            raise ValueError('Register a target before tracking')
        try:
            if self.two_d_tracker is not None:
                box = self.two_d_tracker.track(frame.rgb)
                if box[2] > 0 and box[3] > 0:
                    self._update_from_bbox(frame.K, box)
            if self.kf is not None and self.kf_mean is not None:
                self.kf_mean, self.kf_cov = self.kf.predict(self.kf_mean, self.kf_cov)
            pose = self.est.track_one(rgb=frame.rgb, depth=frame.depth, K=frame.K,
                                      iteration=self.track_iterations)
            if self.kf is not None and self.kf_mean is not None:
                self.kf_mean, self.kf_cov = self.kf.update(self.kf_mean, self.kf_cov, self._pose6(pose))
            if self.est.pose_last is None or not np.isfinite(pose).all():
                raise ValueError('Pose registration/tracking failed')
        except Exception:
            # track_one() only overwrites pose_last on success, so a raise from
            # inside this block usually still leaves the last good pose in place --
            # worth keeping as a relock seed. A pose that *did* come back non-finite
            # (the isfinite check above) is pose_last itself now, so don't keep it.
            keep = (self.est.pose_last is not None
                   and np.isfinite(self.est.pose_last.detach().cpu().numpy()).all())
            self.reset(keep_prior=keep)
            raise
        self._frames_since_score += 1
        if self.drift_score_ratio <= 0 or self._frames_since_score < self.score_interval:
            return pose
        self._frames_since_score = 0
        self.last_score = self._score(frame)
        if not np.isfinite(self.last_score):
            self.reset()
            raise ValueError('Pose score is not finite')
        # Rolling baseline (not a fixed registration-time value) absorbs legitimate
        # viewing-angle score shifts; a rejected score is excluded from the blend so
        # the baseline can't chase its own drift down.
        drift_threshold = (self.score_ema
                           - abs(self.score_ema) * (1 - self.drift_score_ratio))
        self.last_score_ok = not (self.drift_score_ratio > 0 and self.last_score < drift_threshold)
        if self.last_score_ok:
            self.score_ema = (self.score_ema_alpha * self.last_score
                              + (1 - self.score_ema_alpha) * self.score_ema)
        print(f"[SCORE] frame={getattr(frame, 'identifier', '?')} mode=track "
              f"score={self.last_score:.4f} registration_baseline={self.registration_score:.4f} "
              f"ema_after={self.score_ema:.4f} threshold={drift_threshold:.4f} "
              f"enabled={self.drift_score_ratio > 0} "
              f"decision={'PASS' if self.last_score_ok else 'SOFT_LOW'}", flush=True)
        return pose

    def _finalize_lock(self, frame, mask, pose):
        """Shared post-lock step for register()/relock() successes (both the
        estimate() mask branch and Tier-0 Cutie relock): seed Cutie/Kalman from
        the mask and set the score baseline future frames are compared against."""
        if self.two_d_tracker is not None:
            self.two_d_tracker.initialize(frame.rgb, {'mask': mask.astype(np.uint8)})
        if self.kf is not None:
            self.kf_mean, self.kf_cov = self.kf.initiate(self._pose6(pose))
        self.last_score = self._score(frame)
        if not np.isfinite(self.last_score):
            self.reset()
            raise ValueError('Pose score is not finite')
        self.score_ema = self.last_score
        self.registration_score = self.last_score
        self.last_score_ok = True
        self._frames_since_score = 0
        print(f"[SCORE] frame={getattr(frame, 'identifier', '?')} mode=register "
              f"score={self.last_score:.4f} registration_baseline={self.registration_score:.4f} "
              f"ema_after={self.score_ema:.4f}", flush=True)
        return pose

    @staticmethod
    def _pose6(pose):
        p = np.asarray(pose).reshape(4, 4)
        return np.r_[p[:3, 3], Rotation.from_matrix(p[:3, :3]).as_euler('xyz')]

    def _update_from_bbox(self, K, box):
        p = self.est.pose_last.detach().cpu().numpy().reshape(4, 4)
        z = float(p[2, 3])
        u, v = box[0] + box[2] / 2, box[1] + box[3] / 2
        xy = np.array([(u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1]])
        self.kf_mean, self.kf_cov = self.kf.update_from_xy(self.kf_mean, self.kf_cov, xy)
        p[:3, 3] = self.kf_mean[:3]
        p[:3, :3] = Rotation.from_euler('xyz', self.kf_mean[3:6]).as_matrix()
        import torch
        self.est.pose_last = torch.from_numpy(p.astype(np.float32)).to(self.est.pose_last.device).unsqueeze(0)

    def validate_geometry(self, frame, pose, box=None):
        """Reject off-image, depth-inconsistent, or detector-disagreeing centers.

        This is a conservative position guard, not a rotation/identity metric.
        """
        center = (pose @ np.linalg.inv(self.to_origin))[:3, 3]
        if not np.isfinite(center).all() or center[2] <= 0:
            raise ValueError('Lost: invalid object center')
        projected = frame.K @ center
        u, v = projected[:2] / projected[2]
        h, w = frame.depth.shape
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError('Lost: projected center outside image')
        if box is not None:
            x1, y1, x2, y2 = box
            mx, my = self.detector_bbox_margin * (x2-x1), self.detector_bbox_margin * (y2-y1)
            if not (x1-mx <= u <= x2+mx and y1-my <= v <= y2+my):
                raise ValueError('Lost: pose center disagrees with detector bbox')
        radius = np.linalg.norm(self.bbox[1] - self.bbox[0]) * 0.5
        # Sample depth over a chunk of the object's own projected size, not a fixed
        # pixel patch: a fixed window is tiny next to a near object -- easy to land
        # entirely inside a glossy-surface specular depth dropout -- and can spill
        # into the background next to a far one.
        focal = (frame.K[0, 0] + frame.K[1, 1]) / 2
        px_radius = max(3, int(0.4 * focal * radius / center[2]))
        x, y = int(u), int(v)
        patch = frame.depth[max(0,y-px_radius):min(h,y+px_radius+1),
                            max(0,x-px_radius):min(w,x+px_radius+1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size < 5:
            raise ValueError('Lost: insufficient depth near pose center')
        if abs(float(np.median(valid)) - center[2]) > radius + 0.025:
            raise ValueError('Lost: pose depth disagrees with camera depth')

    def _relock(self, frame, mask, prior_pose):
        """Cheap single-hypothesis re-lock: keep prior_pose's orientation, refresh
        translation from the fresh mask, and refine with track_one()'s local
        refiner instead of register()'s ~250-hypothesis global orientation search.
        Returns None (caller falls back to register()) on any failure -- a bad
        relock must never reach the caller as a false TRACKING state.
        """
        import torch
        translation = self.est.guess_translation(depth=frame.depth, mask=mask.astype(np.uint8), K=frame.K)
        if not np.isfinite(translation).all() or not translation.any():
            return None
        seeded = prior_pose.reshape(1, 4, 4).clone()
        seeded[0, :3, 3] = torch.as_tensor(translation, device=seeded.device, dtype=seeded.dtype)
        self.est.pose_last = seeded
        try:
            pose = self.est.track_one(rgb=frame.rgb, depth=frame.depth, K=frame.K,
                                      iteration=self.relock_iterations)
        except Exception:
            self.est.pose_last = None
            return None
        if pose is None or not np.isfinite(pose).all():
            self.est.pose_last = None
            return None
        try:
            self.validate_geometry(frame, pose)
        except ValueError:
            self.est.pose_last = None
            return None
        return pose

    def project_bbox(self, K, pose):
        """2D bbox of the object's 3D extent at `pose`; used to limit a fresh
        detector search to roughly where the object currently is instead of the
        full frame."""
        centered_pose = pose @ np.linalg.inv(self.to_origin)
        mins, maxs = self.bbox
        corners = np.array(list(itertools.product(*zip(mins, maxs))))
        corners_h = np.hstack([corners, np.ones((8, 1))])
        cam_pts = (centered_pose @ corners_h.T).T[:, :3]
        if np.any(cam_pts[:, 2] <= 0):
            return None
        proj = (K @ cam_pts.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]

    def cutie_probe(self, frame):
        """Advance Cutie's VOS state through a FoundationPose loss and report
        whether it still has a rough lock, without touching FoundationPose state.
        A live Cutie lock lets recovery skip the heavy detector entirely. Requires
        a prior pose: that's set (via reset(keep_prior=True)) exactly when Cutie
        was last initialized with a target, so it also means "Cutie has memory to
        track from" -- calling track() before any initialize() crashes Cutie."""
        if (self.two_d_tracker is None or self._prior_pose is None
                or self.relock_iterations <= 0):
            return None
        box = self.two_d_tracker.track(frame.rgb)
        if box[2] <= 0 or box[3] <= 0:
            return None
        return box

    def mask_from_bbox(self, frame, bbox_xywh):
        """Coarse proxy mask (rectangle x valid depth) for _relock() when only a
        Cutie bbox is available -- good enough for guess_translation()'s
        depth-weighted centroid, without a detector call."""
        x, y, w, h = bbox_xywh
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2 = min(frame.depth.shape[1], int(x + w))
        y2 = min(frame.depth.shape[0], int(y + h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        mask = np.zeros(frame.depth.shape, dtype=bool)
        mask[y1:y2, x1:x2] = frame.depth[y1:y2, x1:x2] > 0
        return mask if mask.any() else None

    def relock_from_bbox(self, frame, bbox_xywh):
        """Tier-0 recovery entry point: relock straight from a Cutie bbox, with
        no detector call at all. None if there's no prior orientation to seed
        from, or the relock doesn't pass validate_geometry. On success this runs
        the same Cutie/Kalman-reinit + score-baseline bookkeeping a register()
        success gets, via _finalize_lock -- otherwise the next tracking frame's
        drift check would compare against a stale/absent score baseline."""
        if self._prior_pose is None:
            return None
        mask = self.mask_from_bbox(frame, bbox_xywh)
        if mask is None:
            return None
        pose = self._relock(frame, mask, self._prior_pose)
        if pose is None:
            return None
        return self._finalize_lock(frame, mask, pose)

    def draw(self, rgb, K, pose):
        from Utils import draw_posed_3d_box, draw_xyz_axis
        centered_pose = pose @ np.linalg.inv(self.to_origin)
        image = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=centered_pose, bbox=self.bbox)
        return draw_xyz_axis(image, ob_in_cam=pose, scale=0.05, K=K,
                             thickness=2, transparency=0, is_input_rgb=True)


# ---------------------------------------------------------------- recovery state machine


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

            # Soft signals below: pose exists, but confidence/geometry/detector disagree.
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
                        hard_reason = f'Lost: detector missed target {self.loss_patience} times'
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
            # Per-stage detector timings (e.g. a two-stage detector) print only when present.
            stages = ''.join(f" {k}={v}" for k, v in timings.items()
                             if k.endswith('_ms') and k != 'latency_ms' and v is not None)
            print(f"[SEARCH_END] frame={getattr(frame, 'identifier', '?')} "
                  f"attempt={self.search_attempt} retry={self.search_attempt-1} source={source}"
                  f"{stages} "
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
