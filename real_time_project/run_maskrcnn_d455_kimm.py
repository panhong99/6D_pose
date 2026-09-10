"""Automatic bbox + mask preview. S saves results; Q quits. No manual ROI."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time


def save_result(root, rgb, detections, preview):
    import cv2
    folder = root / f'{time.time_ns()}_kimm'
    folder.mkdir(parents=True)
    images = {'rgb_kimm.png': cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
              'detections_kimm.png': cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)}
    records = []
    for index, detection in enumerate(detections):
        name = f'mask_{index:03d}_kimm.png'
        images[name] = detection['mask'].astype('uint8') * 255
        records.append(dict((k, v) for k, v in detection.items() if k != 'mask'))
        records[-1]['mask_file'] = name
    for name, image in images.items():
        if not cv2.imwrite(str(folder / name), image):
            raise OSError(f'Could not save {name}')
    (folder / 'detections_kimm.json').write_text(json.dumps(records, indent=2))
    print(f'Saved: {folder}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--serial', default='')
    parser.add_argument('--image', type=Path, help='Test a saved RGB image instead of opening D455')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'debug_maskrcnn_kimm')
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error('score_threshold must be within 0..1')
    if Path('/usr/share/fonts/truetype/dejavu').is_dir():
        os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')
    import cv2
    from FoundationPose.real_time_project.maskrcnn_detector_kimm import MaskRCNNDetector, draw_detections
    detector = MaskRCNNDetector(args.device, args.score_threshold)
    print('COCO pretrained model: Rubik cube is not a trained class. All detections are displayed.', flush=True)
    if args.image:
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            parser.error(f'Cannot read {args.image}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detector.predict(rgb)
        save_result(args.output_dir, rgb, detections, draw_detections(rgb, detections))
        print([(d['label'], round(d['score'], 3)) for d in detections])
        return
    from FoundationPose.real_time_project.d455_source_kimm import D455Source
    with ExitStack() as resources:
        camera = D455Source(serial=args.serial)
        resources.callback(camera.close)
        resources.callback(cv2.destroyAllWindows)
        while True:
            frame = camera.read()
            start = time.perf_counter()
            detections = detector.predict(frame.rgb)
            preview = draw_detections(frame.rgb, detections)
            display = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f'{len(detections)} objects | {time.perf_counter()-start:.2f}s | S:save Q:quit',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Mask R-CNN detection + segmentation kimm', display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                save_result(args.output_dir, frame.rgb, detections, preview)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
