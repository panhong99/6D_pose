"""Sequential YOLOE vs Grounding DINO + SAM2 benchmark. See README.md.

python benchmark.py --input_dir ./test_images --output_dir ./results
"""
import argparse
import csv
import gc
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from pipelines.artifacts import draw_result, side_by_side, write_mask, write_rgb
from pipelines.base import resolve_device, synchronize

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm'}
PIPELINES = ('yoloe', 'grounding_dino_sam2')
DEFAULT_PROMPTS = ["rubik's cube", 'colorful cube', 'toy cube']


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def load_tags(path):
    """Keys use source relative to input_dir; a blank frame_index tags the source."""
    if path is None:
        return {}
    tags = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'source', 'frame_index', 'tag'}.issubset(reader.fieldnames or []):
            raise ValueError('tags_csv needs columns: source,frame_index,tag')
        for line_number, row in enumerate(reader, 2):
            source = row['source'].strip().replace('\\', '/')
            source_path = Path(source)
            if not source or source_path.is_absolute() or '..' in source_path.parts:
                raise ValueError(f'tags_csv line {line_number}: source must be a relative path')
            frame_index = int(row['frame_index']) if row['frame_index'].strip() else None
            if frame_index is not None and frame_index < 0:
                raise ValueError(f'tags_csv line {line_number}: frame_index must be >= 0')
            tag = row['tag'].strip().lower()
            if tag not in ('blur', 'normal'):
                raise ValueError(f'tags_csv line {line_number}: tag must be blur or normal')
            key = (source_path.as_posix(), frame_index)
            if key in tags:
                raise ValueError(f'Duplicate tag for {key}')
            tags[key] = tag
    return tags


def frame_tag(source, frame_index, blur_tags, tags):
    if (source, frame_index) in tags:
        return tags[(source, frame_index)]
    if (source, None) in tags:
        return tags[(source, None)]
    return 'blur' if any(tag.lower() in Path(source).name.lower() for tag in blur_tags) else 'normal'


def collect_frames(args, tags):
    """Decode once to PNG on disk, so both pipelines receive identical RGB pixels."""
    output = args.output_dir.resolve()
    if args.video:
        sources = [args.video.resolve()]
        root = args.video.resolve().parent
    else:
        root = args.input_dir.resolve()
        sources = sorted(path for path in root.rglob('*')
                         if path.is_file() and not path.resolve().is_relative_to(output)
                         and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    if not sources:
        raise ValueError('No supported images or videos found in input')
    frames, errors = [], []

    def add_frame(bgr, source, frame_index=None, timestamp_ms=None):
        frame_id = f'{len(frames):06d}'
        cache_path = Path('frames') / f'{frame_id}.png'
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        write_rgb(output / cache_path, rgb)
        frames.append(dict(frame_id=frame_id, source=source, frame_index=frame_index,
                           timestamp_ms=timestamp_ms,
                           tag=frame_tag(source, frame_index, args.blur_tags, tags),
                           cache_path=cache_path.as_posix(), height=rgb.shape[0], width=rgb.shape[1]))

    for path in sources:
        if args.max_frames and len(frames) >= args.max_frames:
            break
        source = path.relative_to(root).as_posix()
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                errors.append(dict(source=source, error='Cannot decode image'))
                continue
            add_frame(bgr, source)
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                errors.append(dict(source=source, error='Cannot open video'))
                continue
            fps = capture.get(cv2.CAP_PROP_FPS)
            reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            reported_count = int(reported_count) if np.isfinite(reported_count) else 0
            frame_index = 0
            while not args.max_frames or len(frames) < args.max_frames:
                ok, bgr = capture.read()
                if not ok:
                    if frame_index == 0 or (reported_count > 0 and frame_index < reported_count):
                        errors.append(dict(source=source, frame_index=frame_index,
                                           error='Video decoding stopped before reported end'))
                    break
                if frame_index % args.video_stride == 0:
                    timestamp = frame_index * 1000.0 / fps if np.isfinite(fps) and fps > 0 else None
                    add_frame(bgr, source, frame_index, timestamp)
                frame_index += 1
        finally:
            capture.release()
    write_json(output / 'input_errors.json', errors)
    write_json(output / 'frames.json', frames)
    if not frames:
        raise ValueError('No frames could be decoded; see input_errors.json')
    return frames, errors


def read_frame(output, frame):
    path = output / frame['cache_path']
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f'Cannot read cached frame: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_pipeline(name, args):
    # Lazy imports allow either pipeline to run independently of the other stack.
    if name == 'yoloe':
        from pipelines.pipeline_yoloe import YOLOEPipeline
        return YOLOEPipeline(checkpoint=args.yoloe_checkpoint, device=args.device,
                             conf=args.yoloe_conf, imgsz=args.imgsz)
    from pipelines.pipeline_grounding_dino_sam2 import GroundingDINOSAM2Pipeline
    return GroundingDINOSAM2Pipeline(
        sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
        sam2_model=args.sam2_model, dino_model=args.dino_model,
        device=args.device, box_threshold=args.box_threshold, text_threshold=args.text_threshold)


def result_row(output, frame, prompt, prompt_id, pipeline, result, error=None):
    mask_path = None
    if result['mask'] is not None:
        mask_path = f'masks/{pipeline}/{prompt_id}/{frame["frame_id"]}.png'
        write_mask(output / mask_path, result['mask'])
    overlay_path = f'overlays/{pipeline}/{prompt_id}/{frame["frame_id"]}.png'
    rgb = read_frame(output, frame)
    write_rgb(output / overlay_path, draw_result(rgb, {**result, 'error': error},
                                               f'{pipeline} | {prompt}'))
    return dict(frame_id=frame['frame_id'], source=frame['source'],
                frame_index=frame['frame_index'], timestamp_ms=frame['timestamp_ms'],
                tag=frame['tag'], pipeline=pipeline, prompt_id=prompt_id, prompt=prompt,
                success=bool(result['success']), detection_success=result['bbox'] is not None,
                bbox=result['bbox'], has_bbox=result['bbox'] is not None,
                has_mask=result['mask'] is not None, confidence=float(result['confidence']),
                latency_ms=result['latency_ms'], error=error,
                mask_path=mask_path, overlay_path=overlay_path)


def check_result(result, frame):
    if not {'success', 'bbox', 'mask', 'confidence', 'latency_ms'}.issubset(result):
        raise ValueError('Pipeline returned an incomplete result')
    box = result['bbox']
    if box is not None:
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('bbox must be finite xyxy')
        if not (0 <= box[0] < box[2] <= frame['width'] and
                0 <= box[1] < box[3] <= frame['height']):
            raise ValueError('bbox must be valid original-image pixel coordinates')
        result['bbox'] = box.tolist()
    mask = result['mask']
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (frame['height'], frame['width']) or not np.isin(mask, [0, 1]).all():
            raise ValueError('mask must be binary with original image dimensions')
        if not mask.any():
            raise ValueError('Empty masks must be represented as None')
    if bool(result['success']) != (box is not None and mask is not None):
        raise ValueError('success requires both a bbox and a nonempty mask')
    if not np.isfinite(result['confidence']) or not 0 <= result['confidence'] <= 1:
        raise ValueError('confidence must be finite and in [0, 1]')
    if not np.isfinite(result['latency_ms']) or result['latency_ms'] < 0:
        raise ValueError('latency_ms must be finite and nonnegative')


def latency_stats(rows, prefix=''):
    values = [row['latency_ms'] for row in rows
              if not row['error'] and row['latency_ms'] is not None]
    return {f'{prefix}latency_count': len(values),
            f'{prefix}latency_mean_ms': float(np.mean(values)) if values else None,
            f'{prefix}latency_p50_ms': float(np.percentile(values, 50)) if values else None,
            f'{prefix}latency_p95_ms': float(np.percentile(values, 95)) if values else None}


def summarize(rows):
    summaries = []
    groups = dict.fromkeys((row['pipeline'], row['prompt_id'], row['prompt']) for row in rows)
    for pipeline, prompt_id, prompt in groups:
        selected = [row for row in rows if row['pipeline'] == pipeline and row['prompt_id'] == prompt_id]
        for tag in ('all', 'blur', 'normal'):
            group = [row for row in selected if tag == 'all' or row['tag'] == tag]
            total = len(group)
            detected = sum(row['detection_success'] for row in group)
            succeeded = sum(row['success'] for row in group)
            summaries.append(dict(pipeline=pipeline, prompt_id=prompt_id, prompt=prompt, tag=tag,
                                  num_frames=total, error_count=sum(bool(row['error']) for row in group),
                                  detection_success_count=detected,
                                  detection_success_rate=detected / total if total else None,
                                  success_count=succeeded, success_rate=succeeded / total if total else None,
                                  bbox_count=sum(row['has_bbox'] for row in group),
                                  mask_count=sum(row['has_mask'] for row in group),
                                  **latency_stats(group),
                                  **latency_stats([row for row in group if row['success']], 'successful_')))
    return summaries


def save_comparisons(output, rows, pipeline_names):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['prompt_id'], row['frame_id']), {})[row['pipeline']] = row
    for (prompt_id, frame_id), records in grouped.items():
        images = []
        for name in pipeline_names:
            bgr = cv2.imread(str(output / records[name]['overlay_path']))
            if bgr is None:
                raise OSError('Could not read generated overlay')
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        write_rgb(output / 'comparisons' / prompt_id / f'{frame_id}.png', side_by_side(images))


def run_benchmark(args, pipeline_factory=None):
    """Run and export. A factory argument lets tests exercise the real I/O loop."""
    pipeline_factory = pipeline_factory or create_pipeline
    output = args.output_dir
    tags = load_tags(args.tags_csv)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output_dir must be new or empty to prevent mixing benchmark runs')
    if args.input_dir and output.resolve() == args.input_dir.resolve():
        raise ValueError('output_dir must differ from input_dir')
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    packages = {}
    for package in ('torch', 'torchvision', 'ultralytics', 'transformers', 'sam2', 'SAM-2', 'numpy', 'opencv-python'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    config = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  python=platform.python_version(), platform=platform.platform(), packages=packages,
                  resolved_device=str(device), cuda_available=torch.cuda.is_available(),
                  gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  precision='float32', setup={},
                  latency_definition='Synchronized infer wall time; excludes disk I/O, model load and configured warmup.',
                  success_definition='Valid bbox and nonempty mask; no ground-truth accuracy evaluation.')
    write_json(output / 'run_config.json', config)
    frames, input_errors = collect_frames(args, tags)
    print(f'{len(frames)} frames, {len(args.prompts)} prompts, device={device}', flush=True)
    rows = []
    had_errors = bool(input_errors)
    for name in args.pipelines:
        pipeline = None
        setup = dict(load_ms=None, error=None, prompts=[])
        config['setup'][name] = setup
        print(f'Loading {name} ...', flush=True)
        try:
            started = time.perf_counter()
            pipeline = pipeline_factory(name, args)
            synchronize(device)
            setup['load_ms'] = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            setup['error'] = f'{type(exc).__name__}: {exc}'
            had_errors = True
            print(f'{name} setup error: {setup["error"]}', file=sys.stderr, flush=True)
        try:
            for prompt_index, prompt in enumerate(args.prompts):
                prompt_id = f'prompt_{prompt_index:03d}'
                warmup = dict(prompt_id=prompt_id, prompt=prompt, prepare_ms=0.0,
                              warmup_ms=0.0, warmup_iterations=args.warmup, error=None)
                setup['prompts'].append(warmup)
                if pipeline is not None and args.warmup:
                    rgb = read_frame(output, frames[0])
                    try:
                        synchronize(device)
                        started = time.perf_counter()
                        pipeline.prepare_prompt(prompt)
                        synchronize(device)
                        warmup['prepare_ms'] = (time.perf_counter() - started) * 1000.0
                        started = time.perf_counter()
                        for _ in range(args.warmup):
                            pipeline.infer(rgb, prompt)
                        synchronize(device)
                        warmup['warmup_ms'] = (time.perf_counter() - started) * 1000.0
                    except Exception as exc:
                        warmup['error'] = f'{type(exc).__name__}: {exc}'
                        had_errors = True
                        print(f'{name} warmup error: {warmup["error"]}', file=sys.stderr, flush=True)
                print(f'{name} | {prompt}', flush=True)
                for frame in frames:
                    error = setup['error']
                    result = dict(success=False, bbox=None, mask=None, confidence=0.0, latency_ms=None)
                    if pipeline is not None:
                        try:
                            inferred = pipeline.infer(read_frame(output, frame), prompt)
                            check_result(inferred, frame)
                            result = inferred
                        except Exception as exc:
                            error = f'{type(exc).__name__}: {exc}'
                            had_errors = True
                            print(f'{name} frame {frame["frame_id"]}: {error}', file=sys.stderr, flush=True)
                    rows.append(result_row(output, frame, prompt, prompt_id, name, result, error))
                # Persist completed groups even if a later run is interrupted.
                write_json(output / 'per_frame.json', rows)
                write_csv(output / 'per_frame.csv', rows)
                write_json(output / 'run_config.json', config)
        finally:
            del pipeline
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    summary = summarize(rows)
    write_json(output / 'summary.json', summary)
    write_csv(output / 'summary.csv', summary)
    save_comparisons(output, rows, args.pipelines)
    config['finished_utc'] = datetime.now(timezone.utc).isoformat()
    config['status'] = 'completed_with_errors' if had_errors else 'completed'
    write_json(output / 'run_config.json', config)
    for row in summary:
        if row['num_frames']:
            latency = row['latency_mean_ms']
            timing = f'{latency:.1f} ms' if latency is not None else 'n/a'
            print(f"{row['pipeline']} | {row['prompt']} | {row['tag']}: "
                  f"success={row['success_count']}/{row['num_frames']} "
                  f"({row['success_rate']:.1%}), latency={timing}, errors={row['error_count']}")
    print(f'Results: {output.resolve()}', flush=True)
    return 1 if had_errors else 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input_dir', type=Path, help='Recursively read images and short videos')
    inputs.add_argument('--video', type=Path, help='Read a single video')
    parser.add_argument('--output_dir', type=Path, required=True, help='New or empty directory')
    parser.add_argument('--prompt', action='append', dest='prompts', help='Repeat for independent prompt trials')
    parser.add_argument('--pipelines', nargs='+', choices=PIPELINES, default=list(PIPELINES))
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:N')
    parser.add_argument('--warmup', type=int, default=1, help='Untimed iterations per pipeline/prompt; 0 includes cold start')
    parser.add_argument('--video_stride', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=0, help='Global sampled frame limit; 0 means all')
    parser.add_argument('--blur_tag', action='append', dest='blur_tags', help='Case-insensitive filename marker; default blur')
    parser.add_argument('--tags_csv', type=Path, help='Optional source,frame_index,tag CSV; tag is blur or normal')
    parser.add_argument('--yoloe_checkpoint', default='yoloe-11s-seg.pt')
    parser.add_argument('--yoloe_conf', type=float, default=0.15)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--dino_model', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--sam2_model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--sam2_checkpoint', type=Path)
    parser.add_argument('--sam2_config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.prompts = list(dict.fromkeys(args.prompts or DEFAULT_PROMPTS))
    args.pipelines = list(dict.fromkeys(args.pipelines))
    args.blur_tags = args.blur_tags or ['blur']
    if any(not prompt.strip().strip('.') for prompt in args.prompts):
        parser.error('--prompt must be nonempty')
    if any(not tag.strip() for tag in args.blur_tags):
        parser.error('--blur_tag must be nonempty')
    if args.warmup < 0 or args.video_stride < 1 or args.max_frames < 0 or args.imgsz < 1:
        parser.error('warmup/max_frames must be >= 0 and video_stride/imgsz must be > 0')
    for name in ('yoloe_conf', 'box_threshold', 'text_threshold'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'--{name} must be in [0, 1]')
    if args.input_dir and not args.input_dir.is_dir():
        parser.error('--input_dir must be an existing directory')
    if args.video and (not args.video.is_file() or args.video.suffix.lower() not in VIDEO_EXTENSIONS):
        parser.error('--video must be an existing supported video file')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_benchmark(args)
    except (ValueError, OSError) as exc:
        print(f'Benchmark error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
