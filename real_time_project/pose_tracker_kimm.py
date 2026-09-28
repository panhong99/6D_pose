"""CAD loading and FoundationPose state; independent of SAM, camera and UDP."""
import itertools
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


class PoseTracker:
    def __init__(self, mesh_file, mesh_scale, debug_dir, register_iterations=5,
                 track_iterations=2, drift_score_ratio=0.3, detector_bbox_margin=0.5,
                 relock_iterations=6):
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
            plus_src = '/home/panhong/pan/FoundationPose-plus-plus/src'
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
        self.kf_mean = self.kf_cov = None
        self.kf_mean = self.kf_cov = None

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
        prior_pose = self._prior_pose
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
                raise ValueError('Lost: pose center disagrees with DINO bbox')
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
        if self.two_d_tracker is None or self._prior_pose is None:
            return None
        box = self.two_d_tracker.track(frame.rgb)
        if box[2] <= 0 or box[3] <= 0:
            return None
        return box

    def mask_from_bbox(self, frame, bbox_xywh):
        """Coarse proxy mask (rectangle x valid depth) for _relock() when only a
        Cutie bbox is available -- good enough for guess_translation()'s
        depth-weighted centroid, without a detector+SAM2 call."""
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
