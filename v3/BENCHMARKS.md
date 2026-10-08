# Local inference benchmark

Measured on 7 October 2026 using the same first 20 frames of the 1920x1080
Pexels meadow clip, YOLO11x at 640x640, confidence 0.10, batch size 1, and the
same v3 detection pipeline.

Hardware:

- Intel Core i7-1355U, 10 cores / 12 logical processors
- Intel Iris Xe integrated GPU
- No CUDA device

| Backend | First frame | Mean frames 2-20 | Steady FPS | Frames detecting drone |
|---|---:|---:|---:|---:|
| PyTorch CPU | 3040.2 ms | 700.5 ms | 1.43 | 20/20 |
| OpenVINO CPU | 7682.9 ms | 1023.4 ms | 0.98 | 20/20 |
| OpenVINO Intel GPU | 21185.1 ms | 115.9 ms | 8.63 | 20/20 |

The first OpenVINO GPU frame includes one-time model compilation. Do not use an
overall average from a 20-frame cold-start run as the sustained FPS figure.
These figures measure detector inference, not end-to-end camera latency, and
they are specific to this model, input size, machine, drivers, and clip.

Conclusion: use `--device intel:gpu` with the OpenVINO export on this laptop.
OpenVINO CPU is not beneficial here. A smaller drone-trained model is still
needed to move substantially beyond 8-9 detector FPS at 640x640.
