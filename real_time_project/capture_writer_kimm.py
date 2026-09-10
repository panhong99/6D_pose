"""Persist a SAM selection before registration, including rejected masks."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from FoundationPose.real_time_project.mask_viz_kimm import mask_overlay


class CaptureWriter:
    def __init__(self, root, settings):
        self.root = Path(root) / f'captures_{time.time_ns()}_kimm'
        self.root.mkdir(parents=True, exist_ok=False)
        self.settings = settings
        self.count = 0
        print(f'Mask captures: {self.root}', flush=True)

    def save(self, frame, mask, box, accepted):
        self.count += 1
        folder = self.root / f'{self.count:04d}_kimm'
        folder.mkdir()
        mask = np.asarray(mask, dtype=bool)
        overlay = mask_overlay(frame.rgb, mask)
        images = {'rgb_kimm.png': cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                  'mask_kimm.png': mask.astype(np.uint8) * 255,
                  'overlay_kimm.png': cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)}
        for name, image in images.items():
            if not cv2.imwrite(str(folder / name), image):
                raise OSError(f'Failed to save {folder / name}')
        np.save(folder / 'depth_m_kimm.npy', frame.depth)
        np.savetxt(folder / 'K_kimm.txt', frame.K)
        metadata = dict(self.settings, accepted=bool(accepted), box_xyxy=list(box),
                        frame_identifier=frame.identifier, timestamp_ns=frame.timestamp_ns,
                        depth_unit='meter', timestamp_kind='host_receipt',
                        mask_pixels=int(mask.sum()),
                        valid_mask_depth_pixels=int(np.count_nonzero(mask & (frame.depth > 0))))
        (folder / 'metadata_kimm.json').write_text(json.dumps(metadata, indent=2))
        print(f'Saved mask (accepted={accepted}): {folder}', flush=True)
        return folder

    @staticmethod
    def result(folder, pose=None, error=None):
        if pose is not None:
            np.savetxt(folder / 'initial_pose_kimm.txt', np.asarray(pose).reshape(4, 4))
        if error is not None:
            (folder / 'registration_error_kimm.txt').write_text(str(error))
