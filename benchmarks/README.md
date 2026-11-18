# Three Inference Modes

This benchmark evaluates Oxford-IIIT Pet binary foreground/background inference
with the validated Meta DINOv2 ViT-S/14 backbone and the saved smoke-test
segmentation head. Every mode uses the same checkpoint, preprocessing, head,
32-image validation subset, and device-resident 224x224 input.

| Mode | Replaced transformer blocks |
| --- | --- |
| `full_attention` | none |
| `moderate_dwconv` | 2, 5, 8, 11 |
| `aggressive_dwconv` | 1, 2, 3, 4, 5, 6, 7, 8 |

The policies are fixed experimental choices, not optimized replacement sets.
For a DWConv block, the reference patcher exactly transfers the pretrained value
projection and output projection. The depthwise convolution and CLS-mixing
parameters are new. The benchmark validates these transfers, then moves the
complete patched model to CUDA before inference.

Latency uses CUDA events for a device-resident single image and includes backbone
feature extraction plus the segmentation head. It excludes preprocessing and
host-to-device transfer. The largest observed latency is an observation, not a
formal WCET.

Run the benchmark:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_inference_modes.py
```

The command writes `outputs/inference_benchmark/results.json` with block policy,
checkpoint provenance, device, validation mIoU, and median/p95/p99/mean/std/max
observed latency in milliseconds for every mode.
