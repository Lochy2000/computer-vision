from __future__ import annotations

import argparse
from collections.abc import Sequence
from time import perf_counter

from drone_detection.detectors.ultralytics import UltralyticsDetector
from drone_detection.output import AnnotatedVideoWriter, JsonlWriter, annotate_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a detector and write model-independent JSONL results."
    )
    parser.add_argument("--source", required=True, help="Video path, stream URL, or camera ID")
    parser.add_argument("--weights", required=True, help="Local Ultralytics .pt weights")
    parser.add_argument("--output", default="runs/detections.jsonl")
    parser.add_argument("--annotated-video", help="Optional output MP4 with detection boxes")
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--show", action="store_true", help="Show annotated frames while running")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=960,
        help="Width of the live preview window in pixels; does not affect inference or output",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default=None, help="For example: cpu, 0, or cuda:0")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional model class names to retain, e.g. drone bird aircraft",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from drone_detection.pipeline import process_source

    detector = UltralyticsDetector(
        weights=args.weights,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        allowed_classes=set(args.classes) if args.classes else None,
    )

    frames = 0
    total_inference_ms = 0.0
    started = perf_counter()
    video_writer = (
        AnnotatedVideoWriter(args.annotated_video, args.output_fps)
        if args.annotated_video
        else None
    )
    def handle_frame(frame, result) -> None:
        if video_writer is not None:
            video_writer.write(frame, result)
        if args.show:
            import cv2

            rendered = annotate_frame(frame, result)
            if args.preview_width > 0 and rendered.shape[1] > args.preview_width:
                scale = args.preview_width / rendered.shape[1]
                preview_size = (args.preview_width, round(rendered.shape[0] * scale))
                rendered = cv2.resize(rendered, preview_size, interpolation=cv2.INTER_AREA)
            cv2.imshow("Drone Detection v3", rendered)
            cv2.waitKey(1)

    interrupted = False
    try:
        with JsonlWriter(args.output) as writer:
            for result in process_source(
                args.source,
                detector,
                args.max_frames,
                frame_callback=handle_frame if video_writer or args.show else None,
            ):
                writer.write(result)
                frames += 1
                total_inference_ms += result.inference_ms
                if args.progress_every > 0 and frames % args.progress_every == 0:
                    elapsed = perf_counter() - started
                    print(
                        f"frame={frames} detections={len(result.detections)} "
                        f"inference={result.inference_ms:.0f}ms "
                        f"effective_fps={frames / elapsed:.2f}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping cleanly after keyboard interrupt...", flush=True)
    finally:
        if video_writer is not None:
            video_writer.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    average_ms = total_inference_ms / frames if frames else 0.0
    processing_fps = 1000.0 / average_ms if average_ms else 0.0
    print(
        f"Processed {frames} frames; detector mean {average_ms:.2f} ms/frame "
        f"({processing_fps:.2f} FPS); results: {args.output}"
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())

