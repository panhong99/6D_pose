"""CAD loading and FoundationPose state; independent of SAM, camera and UDP."""
import time
from pathlib import Path

import numpy as np
import trimesh


class PoseTracker:
    def __init__(self, mesh_file, mesh_scale, debug_dir, register_iterations=5,
                 track_iterations=2, drift_score_ratio=0.6):
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
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        self.est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(), debug=0, debug_dir=str(debug_dir))
        self.register_iterations = register_iterations
        self.track_iterations = track_iterations
        self.drift_score_ratio = drift_score_ratio
        self.score_ema_alpha = 0.2
        self.score_ema = None
        self.last_score = None
        self.last_score_ok = True
        self.registration_score = None

    def reset(self):
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
        register_ms = None
        started = time.perf_counter()
        try:
            if mask is not None:
                self.reset()
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != frame.depth.shape or np.count_nonzero(mask & (frame.depth > 0)) < 100:
                    raise ValueError('Mask needs at least 100 valid depth pixels')
                register_start = time.perf_counter()
                try:
                    pose = self.est.register(K=frame.K, rgb=frame.rgb, depth=frame.depth,
                                             ob_mask=mask, iteration=self.register_iterations)
                finally:
                    # register returns a CPU pose, synchronizing its GPU work.
                    register_ms = (time.perf_counter() - register_start) * 1000
            else:
                if self.est.pose_last is None:
                    raise ValueError('Register a target before tracking')
                pose = self.est.track_one(rgb=frame.rgb, depth=frame.depth, K=frame.K,
                                          iteration=self.track_iterations)
            # Validate the pose before spending a scorer forward pass on it: a NaN/None
            # pose_last would otherwise crash inside _score() instead of raising cleanly.
            if self.est.pose_last is None or not np.isfinite(pose).all():
                raise ValueError('Pose registration/tracking failed')
        except Exception:
            self.reset()
            raise
        finally:
            if mask is not None:
                print(f"[REGISTER] frame={getattr(frame, 'identifier', '?')} "
                      f"register_ms={register_ms} "
                      f"estimate_total_ms={(time.perf_counter()-started)*1000:.1f}", flush=True)

        self.last_score = self._score(frame)
        if not np.isfinite(self.last_score):
            self.reset()
            raise ValueError('Pose score is not finite')
        if mask is not None:
            self.score_ema = self.last_score
            self.registration_score = self.last_score
            self.last_score_ok = True
            print(f"[SCORE] frame={getattr(frame, 'identifier', '?')} mode=register "
                  f"score={self.last_score:.4f} registration_baseline={self.registration_score:.4f} "
                  f"ema_after={self.score_ema:.4f}", flush=True)
        else:
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
            mx, my = 0.15 * (x2-x1), 0.15 * (y2-y1)
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

    def draw(self, rgb, K, pose):
        from Utils import draw_posed_3d_box, draw_xyz_axis
        centered_pose = pose @ np.linalg.inv(self.to_origin)
        image = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=centered_pose, bbox=self.bbox)
        return draw_xyz_axis(image, ob_in_cam=pose, scale=0.05, K=K,
                             thickness=2, transparency=0, is_input_rgb=True)
