# Semantic Segmentation Baseline

This baseline uses Meta's official pretrained DINOv2 ViT-S/14 checkpoint from
`reference/weights/dinov2_vits14_pretrain.pth` as a frozen feature extractor.
The lightweight head consumes spatial patch features rather than the CLS token.

## Dataset Choice

The first smoke test uses Oxford-IIIT Pet instead of ADE20K. ADE20K requires an
account and acceptance of its terms before download. Oxford-IIIT Pet provides
pixel-level trimaps, has a clear CC BY-SA 4.0 license, and is roughly 800 MB.
Its official page is https://www.robots.ox.ac.uk/~vgg/data/pets/.

Torchvision downloads the dataset to:

```
datasets/
  oxford-iiit-pet/
    images/
    annotations/
      trimaps/
      trainval.txt
      test.txt
```

The source trimap values are `1` for pet foreground, `2` for ambiguous border,
and `3` for background. The baseline maps them to training labels `1`, `255`
(ignore), and `0`. Therefore it reports IoU for `background` and `pet`; border
pixels do not influence loss or metrics.

## Reproduce The Smoke Test

The first command deliberately downloads the dataset. It uses a fixed seed, 64
training images, 32 validation images, 224x224 input, one epoch, batch size 4,
and FP32. It is a pipeline check, not benchmark evidence.

```powershell
.\.venv\Scripts\python.exe baseline\train_segmentation.py --download
```

After the data exists, rerun without `--download`:

```powershell
.\.venv\Scripts\python.exe baseline\train_segmentation.py
```

Artifacts appear in `outputs/pet_smoke/`: `smoke_checkpoint.pt`, `metrics.json`,
and `sample_prediction.png`.

## Evaluate A Saved Checkpoint

This command reloads the saved segmentation head, strictly reloads the official
DINOv2-Small backbone, rebuilds the seeded split, and compares the result with
the saved metric.

```powershell
.\.venv\Scripts\python.exe baseline\train_segmentation.py --mode eval --checkpoint outputs\pet_smoke\smoke_checkpoint.pt --output-dir outputs\pet_smoke\re_evaluation
```

## Small Experimental Run

The default training command uses the same fixed 64/32 source split and runs
three epochs. It evaluates after every epoch, records source image IDs in
`split_manifest.json`, and saves both the best-by-validation-mIoU checkpoint and
the final checkpoint.

```powershell
.\.venv\Scripts\python.exe baseline\train_segmentation.py --output-dir outputs\pet_experiment --epochs 3
```

`metrics.json` records per-epoch loss, validation mIoU, per-class IoU, device,
runtime, and peak GPU memory. `sample_visualization.png` shows input, ground
truth, and prediction from left to right. This remains a small Oxford-IIIT Pet
binary foreground/background experiment, not ADE20K performance.
