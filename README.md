# WMA-YOLO

**Wavelet-Domain Modulation Attention for Lightweight Multi-Class Bridge Defect Detection on Edge Devices**

Code, model configurations, training records and the twelve trained checkpoints of the paper
(IEEE Access, manuscript Access-2026-34194, under review). The bridge20 images are proprietary and are
not distributed; everything else needed to repeat the evaluation is here.

This repository is a fork of [Ultralytics](https://github.com/ultralytics/ultralytics) **8.4.33** and is
released under the same AGPL-3.0 license.

---

## What the paper evaluates

WMA applies attention in the wavelet domain: a 1×1 convolution reduces the channels, a parameter-free Haar
DWT splits the features into LL/LH/HL/HH sub-bands, channel-wise multi-head self-attention and a learned
sigmoid gate modulate the sub-bands, and the inverse transform with a residual connection returns the
output. WMA-YOLO places the module after the C3k2 block at the four backbone stages L2, L4, L6 and L8
(`C3k2_WMA`), on YOLOv11n and on YOLO26n.

The evaluation protocol:

* **Bridge-level split.** 7,037 photographs of 188 bridges, 70:10:20 by bridge: 4,921 training images
  (130 bridges), 705 validation images (21) and 1,411 test images (37). No test bridge contributes a
  training image. Validation only selects the epoch; every reported score is on the test split.
* **Three training runs per core model** (seeds 0, 1, 2) on each detector.
* **Rule for a clear difference** (Eq. 7): `threshold = 2 s sqrt(1/n_a + 1/n_b)`, with `s` the pooled
  standard deviation of the two models. A difference counts as clear only if it exceeds the threshold and
  every run of one model scores above every run of the other.
* **External data:** the public [dacl10k](https://github.com/phiyodr/dacl10k-toolkit) benchmark (zero-shot
  and after fine-tuning) and a public crack-only collection with a de-duplicated split (below).

Results on the bridge20 test split (mean of three runs; latency FP32, 640×640, batch 1):

| Detector | Model | Params (M) | GFLOPs | mAP@50 | mAP@50–95 | x86, 1 thread (ms) | Raspberry Pi 5 (ms / FPS) |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv11n | baseline | 2.59 | 6.5 | 16.0 | 7.2 | 72.5 | 138.6 / 7.22 |
| YOLOv11n | + WMA | 3.19 | 7.5 | 16.7 | 7.4 | 89.7 | 202.2 / 4.94 |
| YOLO26n | baseline | 2.41 | 5.7 | 15.2 | 7.0 | 70.7 | 116.2 / 8.61 |
| YOLO26n | + WMA | 3.00 | 6.6 | 13.5 | 6.4 | 88.8 | 179.0 / 5.59 |

All four accuracy differences lie within run-to-run variation. On YOLOv11n, WMA-YOLO raises precision
from 28.8% to 32.8% at the same recall (14.4%) at a confidence threshold of 0.25. The paper gives the
per-class results, the placement and attention comparisons, the branch-activity measurements and the
limitations.

---

## Repository layout

| Path | Content |
|---|---|
| `ultralytics/nn/modules/conv.py` | `WaveletAttention` (WMA), `C3k_WMA`, `C3k2_WMA`, Haar DWT/IDWT |
| `ultralytics/nn/modules/extra_attn_blocks.py` | attention modules used for comparison (CBAM, SE, GCT, …) |
| `ultralytics/cfg/models/26/` | model YAMLs of every variant in Tables 3 and 8 (`yolo26_*`, `yolo11n_*`) |
| `weights/` | the twelve checkpoints of Table 3 (four models × three training runs) |
| `runs/` | `args.yaml` and `results.csv` of every training run reported in the paper |
| `data/bridge20.yaml` | class names of the 20-class task |
| `data/bridge20_splits/` | image identifiers of the three splits and the split report |
| `data/crack_only_dedup/` | de-duplicated split of the public crack-only collection |
| `scripts/make_bridge20_split.py` | builds the bridge-level split from the bridge identifiers |
| `scripts/cpu_latency.py` | x86 single-thread latency |
| `scripts/export_onnx.py`, `scripts/pi_bench.py` | ONNX export and Raspberry Pi 5 timing |

### Checkpoints

| File | Detector | Model | Config |
|---|---|---|---|
| `yolov11n_baseline_run{0,1,2}.pt` | YOLOv11n | baseline | `yolo11n.yaml` (stock) |
| `yolov11n_wma_run{0,1,2}.pt` | YOLOv11n | + WMA | `yolo11n_A4_WMA.yaml` |
| `yolo26n_baseline_run{0,1,2}.pt` | YOLO26n | baseline | `yolo26_A0_baseline.yaml` |
| `yolo26n_wma_run{0,1,2}.pt` | YOLO26n | + WMA | `yolo26_A4_bb4only.yaml` |

`runN` is the training run with seed N. Class 5 is stored as `rebar corrosion` in the checkpoints and is
called `rebar exposure` in the paper.

### Run records

| Folder | Paper | Runs |
|---|---|---|
| `runs/bridge20_core/` | Table 3 | baseline and WMA, both detectors, three runs each (`A0`/`A4` = YOLO26n, `v11n`/`v11n_WMA` = YOLOv11n) |
| `runs/bridge20_placement_attention/` | Table 8 | placements A1–A3, B1, B2, C1 and SE, CBAM, GCT, one run each |
| `runs/bridge20_coco_init/` | Table 10 | COCO-initialized training, three runs each |
| `runs/dacl10k_finetune/` | Table 7 | 200-epoch fine-tuning on dacl10k, three runs each |
| `runs/crack_only_finetune/` | Table 6 | 100-epoch fine-tuning on the de-duplicated crack-only split |

---

## Usage

```bash
git clone https://github.com/wows84570356-ops/WMA-YOLO.git
cd WMA-YOLO
pip install -e .
```

Inference with a released checkpoint:

```python
from ultralytics import YOLO

model = YOLO("weights/yolov11n_wma_run0.pt")
results = model("image.jpg", imgsz=640, conf=0.25)
```

Training with the recipe of the paper (from scratch, 400 epochs; seeds 0, 1 and 2):

```bash
yolo detect train model=ultralytics/cfg/models/26/yolo11n_A4_WMA.yaml data=data/bridge20.yaml \
     epochs=400 patience=0 batch=32 imgsz=640 optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 \
     weight_decay=0.0005 warmup_epochs=3 cos_lr=True close_mosaic=10 box=7.5 cls=0.5 dfl=1.5 \
     pretrained=False seed=0 deterministic=True
```

The exact arguments of every run are in `runs/*/*/args.yaml`. Preprocessing is the Ultralytics letterbox
resize to 640×640; the augmentations are Mosaic (off for the last 10 epochs), horizontal flip, HSV and
scale/translate, without rotation, MixUp or copy-paste.

Rebuilding the split from a folder that holds the source images and labels:

```bash
python scripts/make_bridge20_split.py /path/to/dataset_root
```

Latency:

```bash
python scripts/cpu_latency.py                          # x86, one thread
python scripts/export_onnx.py /path/to/test/images     # builds bundle/ for the Raspberry Pi 5
python3 pi_bench.py                                    # on the Pi, inside bundle/
```

The Pi timing uses a stacked form of the inverse Haar transform; exported as written, the transform becomes
24 scatter operations that ONNX Runtime executes slowly. `export_onnx.py` writes both graphs.

---

## Data

* **bridge20** — photographs from routine inspections, provided by a private company. The images cannot be
  released; `data/bridge20_splits/` lists the image identifiers of each split (the first field of an
  identifier is the bridge), and `split_report.txt` gives the per-class counts.
* **dacl10k** — public; used for zero-shot evaluation (975 validation images) and fine-tuning.
* **Crack-only collection** — five public crack datasets merged. Its public splits place augmented copies of
  one photograph in different splits: 315 of the 1,635 source photographs of the test split (19.3%) also
  appear in its training or validation split. `data/crack_only_dedup/` is the de-duplicated split used for
  fine-tuning (test split unchanged at 1,665 images).

## License

AGPL-3.0, as the upstream Ultralytics package. See `LICENSE`.

## Citation

```bibtex
@article{lin2026wmayolo,
  title   = {{WMA-YOLO}: Wavelet-Domain Modulation Attention for Lightweight Multi-Class Bridge Defect
             Detection on Edge Devices},
  author  = {Lin, Chia-Min and Chiang, Jen-Shiun and Chun, Pang-Jo},
  journal = {IEEE Access},
  note    = {under review},
  year    = {2026}
}
```
