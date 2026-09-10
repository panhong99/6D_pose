"""D455 -> YOLO-World real-time cube detection only. No SAM, no FoundationPose. Q quits."""
import argparse
from contextlib import ExitStack
import time

import cv2

from d455_source_kimm import D455Source


def draw_detections(rgb, boxes, scores, labels):
    image = rgb.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {score:.2f}', (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes', nargs='+', default=['rubiks cube', 'cube'],
                        help='Open-vocabulary text prompts YOLO-World should look for')
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--model', default='yolov8s-world.pt',
                        help='Ultralytics YOLO-World checkpoint name; auto-downloaded on first use')
    parser.add_argument('--serial', default='')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('conf must be within 0..1')
    return args


def main():
    args = parse_args()
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    model.set_classes(args.classes)
    print(f'YOLO-World classes: {args.classes}', flush=True)

    with ExitStack() as resources:
        camera = D455Source(serial=args.serial)
        resources.callback(camera.close)
        resources.callback(cv2.destroyAllWindows)
        while True:
            frame = camera.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            start = time.perf_counter()
            result = model.predict(bgr, conf=args.conf, verbose=False)[0]
            elapsed = time.perf_counter() - start
            boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else []
            scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else []
            labels = ([result.names[int(c)] for c in result.boxes.cls]
                     if len(result.boxes) else [])
            preview = draw_detections(frame.rgb, boxes, scores, labels)
            display = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f'{len(boxes)} detections | {elapsed:.3f}s | Q: quit',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('YOLO-World cube detection kimm', display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
