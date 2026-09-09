# Urban Elements Re-ID Challenge 2026

[![Challenge](https://img.shields.io/badge/ICIP%202026-Grand%20Challenge-blue)](https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org)

Instance-level re-identification of urban objects (containers, crosswalks, rubbish bins, traffic signs) across multiple cameras. This repository contains our solution for the **[ICIP 2026 Urban Elements ReID Grand Challenge](https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026)**, built on top of [Part-Aware Transformer (PAT)](https://github.com/liyuke65535/Part-Aware-Transformer) and extended with DINOv3 fine-tuning, staged training, and per-class k-reciprocal re-ranking.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Setup](#4-setup)
5. [Data Preparation](#5-data-preparation)
6. [Training](#6-training)
   - [PAT Baseline (ViT-B/L)](#61-pat-baseline-vit-bl)
   - [DINOv3 Fine-tuning](#62-dinov3-fine-tuning)
7. [Inference & Submission](#7-inference--submission)
8. [Post-Processing](#8-post-processing)
9. [Local Evaluation](#9-local-evaluation)
10. [Advanced Features](#10-advanced-features)
11. [Configuration Reference](#11-configuration-reference)
12. [Citation & Acknowledgements](#12-citation--acknowledgements)

---

## 1. Problem Overview

| | Detail |
|---|---|
| **Task** | Cross-camera instance re-identification of urban objects |
| **Classes** | `container`, `crosswalk`, `rubbishbins`, `trafficsign` |
| **Train set** | ~17.5 k images · 1,567 IDs · cameras c001–c003, c101–c104 |
| **Val set** | 517 queries (c104) · 1,791 gallery (c101–c103) |
| **Test set (LB)** | 928 queries (c004, **novel camera**) · 2,844 gallery (c001–c003) |
| **Metric** | Mean Average Precision (mAP) |

The key challenge is a **novel test camera** (c004) not seen during training, which drives in the opposite direction — creating a systematic front/back viewpoint inversion versus the training cameras.

---

## 2. Architecture

```
DINOv3 ViT-H+  (840 M params)
  └─ Staged fine-tuning (S1: head only → S2: last N blocks → S3: full)
        └─ ImageNet-normalised input [224 × 112]
              └─ CLS token feature [1280-d]
                    └─ BN bottleneck + ArcFace/CE + Triplet loss
                          └─ k-reciprocal re-ranking (per-class filtered)
```

The pipeline supports two training branches:

| Branch | Backbone | Notes |
|--------|----------|-------|
| **DINOv3 (primary)** | ViT-H+ `lvd1689m` | Staged fine-tuning with ImageNet normalisation |
| PAT baseline | ViT-B/L TransReID | Part-Aware Transformer with CSL and PSD |

---

## 3. Repository Structure

```
Urban-Elements-ReID-Challenge-2026/
│
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── environment.sh              ← One-shot conda environment setup
├── Makefile                    ← Convenience targets (train, eval, etc.)
│
├── train.py                    ← PAT training entry point
├── test.py                     ← PAT evaluation entry point
├── update.py                   ← Generate submission CSV (PAT pipeline)
├── evaluate_csv.py             ← Local mAP/CMC from a prediction CSV
├── postprocess_eval.py         ← Standalone val post-processing pipeline
├── val_viewer.html             ← Interactive validation re-ranking viewer
│
├── config/                     ← YAML configurations (yacs)
│   ├── defaults.py             ← All default hyperparameters
│   ├── PAT.yml                 ← Upstream PAT (Market1501 → DukeMTMC)
│   ├── vit.yml                 ← Plain ViT baseline config
│   ├── UrbanElementsReID_train.yml    ← PAT baseline training
│   ├── UrbanElementsReID_test.yml     ← PAT inference / submission
│   ├── split_v2_train.yml             ← PAT on CSV-driven val split
│   └── UrbanElementsReID_dinov3_*.yml ← Per-experiment DINOv3 configs
│       └── (Hplus · De90 · ADV · B1 · B2 · TSpec · MNNFT · CAdv · PartL …)
│
├── data/
│   ├── build_DG_dataloader.py  ← DataLoader factory
│   ├── build_split_v2.py       ← Build reproducible UAM+Rivas val split
│   ├── datasets/
│   │   ├── UrbanElementsReID.py        ← Training split loader
│   │   ├── UrbanElementsReID_test.py   ← LB test query/gallery
│   │   ├── UrbanElementsReID_val.py    ← Local val query/gallery
│   │   ├── UrbanElementsReID_with_val.py
│   │   ├── UrbanElementsReID_with_MNN.py
│   │   ├── UrbanElementsReID_trafficsign.py  ← Trafficsign specialist
│   │   ├── SplitV2ReID.py              ← CSV-driven split v2 loader
│   │   └── bases.py                    ← BaseDataset ABC
│   ├── samplers/               ← Triplet / softmax samplers
│   └── transforms/             ← Augmentation pipeline
│
├── model/
│   ├── make_model.py           ← Model factory (routes to backbone)
│   └── backbones/
│       ├── vit_pytorch.py      ← PAT ViT with part-attention (796 lines)
│       ├── dinov3_timm.py      ← DINOv3 wrapper with staged training
│       ├── dinov3_part.py      ← Part-aware DINOv3 variant
│       ├── lora.py             ← LoRA adapter module
│       └── resnet*.py · IBN.py ← ResNet backbones
│
├── processor/
│   ├── part_attention_vit_processor.py ← PAT train/eval loop with AMP
│   └── ori_vit_processor_with_amp.py   ← Plain ViT train/eval loop
│
├── loss/                       ← CE · Triplet · ArcFace · Center · patch-memory
├── solver/                     ← Optimizer and LR scheduler factories
│
├── utils/
│   ├── metrics.py              ← R1/mAP evaluation (R1_mAP_eval class)
│   ├── re_ranking.py           ← k-reciprocal re-ranking
│   │                             · re_ranking()              — distance-matrix API
│   │                             · re_ranking_from_features() — feature-vector API
│   ├── caj_re_ranking.py       ← Camera-Aware Jaccard re-ranking
│   └── meter.py · logger.py · file_io.py · comm.py · iotools.py · registry.py
│
├── scripts/                    ← DINOv3 experiment pipeline
│   ├── train_dinov3.py         ← Staged DINOv3 fine-tuning
│   ├── train_dinov3_part.py    ← Part-aware DINOv3 fine-tuning
│   ├── train_cross_encoder.py  ← Cross-encoder training
│   ├── train_offline_msloss.py ← Offline multi-similarity loss training
│   ├── extract_dinov3.py       ← Cache qf/gf features to .npy files
│   ├── extract_dinov3_multires.py  ← Multi-resolution TTA features
│   ├── extract_dinov3_tokens.py    ← Token-level features
│   ├── extract_dinov3_part.py      ← Part features
│   ├── extract_features_allblocks.py ← All-block CLS features
│   ├── extract_train_features.py   ← Training-set features
│   ├── extract_sam3.py             ← SAM3 mask extraction
│   ├── eval_variants.py        ← Pure / rerank / per-class evaluation
│   ├── eval_cls_n.py           ← Last-N CLS block sweep
│   ├── eval_cls_n_concat.py    ← CLS concat evaluation variants
│   ├── eval_cls_n_perclass.py  ← Per-class CLS-N evaluation
│   ├── eval_caj.py             ← Camera-Aware Jaccard evaluation
│   ├── eval_calibrate_adaptive.py  ← Adaptive calibration evaluation
│   ├── eval_ccfm.py            ← CCFM evaluation
│   ├── eval_cross_encoder.py   ← Cross-encoder evaluation
│   ├── eval_cross_encoder_fixed.py
│   ├── eval_offline_msloss.py  ← Offline MS loss evaluation
│   ├── eval_sam3_rerank.py     ← SAM3 geometry-aware re-ranking
│   ├── eval_token_variants.py  ← Token-level experiment evaluation
│   ├── hybrid_perclass.py      ← 2-model per-class routing
│   ├── hybrid_multi_perclass.py ← N-model per-class routing
│   ├── auto_hybrid_best_perclass.py ← Auto-search per-class hybrid
│   ├── cache_ensemble_weighted.py ← Weighted feature-concat ensemble
│   ├── cache_mean_ensemble.py  ← Mean ensemble (for TTA)
│   ├── build_TSpec_hybrid.py   ← Build trafficsign specialist hybrid
│   ├── build_viewer.py         ← Generate val_viewer.html
│   ├── mnn_pseudo_label.py     ← MNN pseudo-label generation
│   ├── submit_fixed_params.py  ← Fixed-parameter submission
│   └── submit_caj_perclass_classparams.py ← CAJ per-class submission
│
└── visualization/              ← PAT attention rollout visualisation
    ├── vit_explain.py
    └── vit_rollout/
        ├── vit_rollout.py
        ├── vit_grad_rollout.py
        └── vit_example.py
```

---

## 4. Setup

### Requirements

- Python ≥ 3.10
- CUDA ≥ 11.8 (GPU training)
- ~40 GB GPU VRAM for DINOv3 ViT-H+ (e.g. A100 40 GB)
- ~16 GB GPU VRAM for PAT ViT-L or DINOv3 ViT-L

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/Urban-Elements-ReID-Challenge-2026.git
cd Urban-Elements-ReID-Challenge-2026

# 2. Create and activate a conda environment
conda create -n urban-reid python=3.10
conda activate urban-reid

# 3. Install dependencies (CUDA 11.8 pinned)
bash environment.sh

# Or for a version-agnostic install:
pip install -r requirements.txt
```

### Pretrained Backbones

| Backbone | Variant | Source |
|---|---|---|
| ViT-B/16 (PAT baseline) | TransReID | [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) |
| DINOv3 ViT-L | `lvd1689m` | `timm` — `dinov3_vitl16_lvd1689m` |
| DINOv3 ViT-H+ | `lvd1689m` | `timm` — `vit_huge_plus_patch16_dinov3.lvd1689m` |

Place downloaded weights under `pretrained/` (gitignored).

---

## 5. Data Preparation

### Download

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026/data) and extract it to a directory of your choice.

### Expected Directory Structure

```
UrbanUAM_Merged/
├── image_train/               ← Training images (labeled, 1,567 IDs)
│   └── <class>_<id>_c<cam>_*.jpg
├── image_query/               ← Test query images  (c004 — LB submission)
├── image_test/                ← Test gallery images (c001–c003 — LB submission)
├── val_image_query/           ← Val query images   (c104 — local evaluation)
├── val_image_test/            ← Val gallery images (c101–c103 — local evaluation)
└── *.csv                      ← Ground-truth class/ID labels
```

### Configure Dataset Path

Open your chosen config file and set `ROOT_DIR`:

```yaml
DATASETS:
  ROOT_DIR: '/path/to/UrbanUAM_Merged'
```

### (Optional) Build a Reproducible Val Split

```bash
python data/build_split_v2.py
```

This generates CSV-based train/val splits for reproducible local evaluation using `SplitV2ReID`.

---

## 6. Training

### 6.1 PAT Baseline (ViT-B/L)

The PAT baseline uses the Part-Aware Transformer with Cross-ID Similarity Learning (CSL) and Part-guided Self-Distillation (PSD).

```bash
python train.py --config_file config/UrbanElementsReID_train.yml

# Or use Makefile
make train CONFIG=config/UrbanElementsReID_train.yml LOG=train.log
```

**Key config fields (`config/UrbanElementsReID_train.yml`):**

```yaml
MODEL:
  NAME: 'part_attention_vit'
  TRANSFORMER_TYPE: 'vit_base_patch16_224_TransReID'
SOLVER:
  OPTIMIZER_NAME: 'SGD'
  MAX_EPOCHS: 60
  BASE_LR: 0.001
  IMS_PER_BATCH: 64
```

Checkpoints are saved every `SOLVER.CHECKPOINT_PERIOD` epochs under `LOG_ROOT/`.

### 6.2 DINOv3 Fine-tuning

The DINOv3 pipeline uses a **3-stage progressive unfreezing** strategy:

| Stage | Epochs | Trainable components | LR scale |
|-------|--------|----------------------|----------|
| S1 — probe | 5 | Classification head only | 1.0× |
| S2 — partial | 15 | Last 6 transformer blocks + head | 0.2× |
| S3 — full | 40 | All parameters | 0.1× |

```bash
# ViT-H+ (requires A100 40 GB)
python scripts/train_dinov3.py \
    --config_file config/UrbanElementsReID_dinov3_Hplus.yml

# ViT-L (fits on 16 GB GPU)
python scripts/train_dinov3.py \
    --config_file config/UrbanElementsReID_dinov3_De90.yml

# Or use Makefile
make train-dinov3 CONFIG=config/UrbanElementsReID_dinov3_Hplus.yml
```

**Key config fields (`config/UrbanElementsReID_dinov3_Hplus.yml`):**

```yaml
MODEL:
  NAME: 'dinov3_vit_large'
  DINOV3_VARIANT: 'vit_huge_plus_patch16_dinov3.lvd1689m'
INPUT:
  SIZE_TRAIN: [224, 112]
  PIXEL_MEAN: [0.485, 0.456, 0.406]   # ImageNet normalisation
  PIXEL_STD:  [0.229, 0.224, 0.225]
SOLVER:
  STAGE1_EPOCHS: 5
  STAGE2_EPOCHS: 15
  STAGE3_EPOCHS: 40
  BASE_LR: 0.0005
  IMS_PER_BATCH: 16                    # Halved for ViT-H+ memory
  CLIP_GRAD_NORM: 5.0                  # Gradient clipping for AMP stability
```

---

## 7. Inference & Submission

### PAT — Generate Submission CSV

```bash
python update.py \
  --config_file config/UrbanElementsReID_test.yml \
  --track outputs/submission
```

Produces `outputs/submission/track_submission.csv` — the top-100 gallery indices per query in Kaggle submission format.

### DINOv3 — Extract Features then Evaluate

```bash
# Step 1: Extract and cache query/gallery features to feat_cache/<tag>/
python scripts/extract_dinov3.py \
  --config_file config/UrbanElementsReID_dinov3_Hplus.yml

# Step 2: Run post-processing variants and write submission CSV
python scripts/eval_variants.py \
  --config_file config/UrbanElementsReID_dinov3_Hplus.yml \
  --tag Hplus

# Or chain both with Makefile
make extract CONFIG=config/UrbanElementsReID_dinov3_Hplus.yml
make eval-variants CONFIG=config/UrbanElementsReID_dinov3_Hplus.yml
```

Features are cached as `feat_cache/<tag>/qf.npy`, `gf.npy`, `meta.json` (gitignored).

---

## 8. Post-Processing

The repository supports several post-processing stages that can be applied after feature extraction:

| Variant | Description |
|---------|-------------|
| Pure cosine | Cosine distance over the full gallery |
| Class filter | Restrict gallery to the query's predicted class |
| Global k-rerank | k-reciprocal re-ranking (default: k1=20, k2=6, λ=0.3) |
| Per-class rerank | Class filter + per-class grid-searched k-rerank |

### Per-Class Re-ranking

Each of the four object classes can use individually tuned re-ranking parameters (k1, k2, λ), searched on the validation set:

```bash
python scripts/eval_variants.py \
  --config_file config/UrbanElementsReID_dinov3_Hplus.yml \
  --tag Hplus \
  --perclass_grid_search
```

### Standalone Post-Processing Evaluation

```bash
python postprocess_eval.py
```

Runs all variants sequentially (pure → class filter → global rerank → per-class rerank) and prints a comparison table against the local val set.

---

## 9. Local Evaluation

Compute mAP and CMC curves on the val set without uploading to Kaggle:

```bash
python evaluate_csv.py \
  --path "/path/to/UrbanUAM_Merged/csv_folder/" \
  --track "outputs/track_submission.csv"

# Or use Makefile
make eval DATA_ROOT=/path/to/UrbanUAM_Merged TRACK=outputs/track_submission.csv
```

Compares `track_submission.csv` against ground-truth CSVs (`query_classes.csv`, `test_classes.csv`) and reports mAP + CMC-k curves.

---

## 10. Advanced Features

### Multi-Resolution TTA

```bash
python scripts/extract_dinov3_multires.py \
  --config_file config/UrbanElementsReID_dinov3_Hplus.yml \
  --scales 192 224 256 288

python scripts/cache_mean_ensemble.py \
  --tags HpTTA_res192 HpTTA_res224 HpTTA_res256 HpTTA_res288
```

### Camera-Aware Jaccard (CAJ) Re-ranking

```bash
python scripts/eval_caj.py \
  --config_file config/UrbanElementsReID_dinov3_Hplus.yml \
  --tag Hplus
```

### Traffic Sign Specialist

Train a specialist model on traffic sign images only, then blend with the general model:

```bash
# Train specialist (warm-start from a general model checkpoint)
python scripts/train_dinov3.py \
  --config_file config/UrbanElementsReID_dinov3_TSpec.yml

# Build hybrid submission
python scripts/build_TSpec_hybrid.py \
  --general_tag Hplus \
  --specialist_tag TSpec
```

### Cross-Model Feature Ensembles

```bash
# Weighted feature-concatenation ensemble
python scripts/cache_ensemble_weighted.py \
  --tags Hplus De90 run3_ep30 \
  --weights 1.0 1.0 1.0

# Per-class model routing (route each class to a chosen model)
python scripts/hybrid_multi_perclass.py \
  --tags Hplus De90 ADV run3_ep30
```

### Interactive Validation Viewer

```bash
# Rebuild the viewer from cached features
python scripts/build_viewer.py --tag Hplus

# Open in browser
open val_viewer.html
```

---

## 11. Configuration Reference

All hyperparameters are managed via [yacs](https://github.com/rbgirshick/yacs). The complete schema with defaults is in `config/defaults.py`.

### Common Fields

```yaml
MODEL:
  NAME: 'dinov3_vit_large'           # Model type
  DINOV3_VARIANT: '...'              # timm model identifier string
  PRETRAIN_PATH: '/path/to/weights'  # Pretrained backbone checkpoint

INPUT:
  SIZE_TRAIN: [224, 112]             # [height, width]
  PIXEL_MEAN: [0.485, 0.456, 0.406] # Use ImageNet stats for DINOv3
  PIXEL_STD:  [0.229, 0.224, 0.225] # Use [0.5,0.5,0.5] for PAT

DATASETS:
  ROOT_DIR: '/path/to/dataset'       # Dataset root directory
  TRAIN: ('UrbanElementsReID',)      # Training dataset class name
  TEST:  ('UrbanElementsReID_val',)  # Eval dataset: _val for local, _test for LB

SOLVER:
  STAGE1_EPOCHS: 5                   # Head-only stage (DINOv3 only)
  STAGE2_EPOCHS: 15                  # Partial-unfreeze stage (DINOv3 only)
  STAGE3_EPOCHS: 40                  # Full fine-tune stage (DINOv3 only)
  MAX_EPOCHS: 60                     # Total training epochs
  BASE_LR: 0.0005                    # Peak learning rate
  IMS_PER_BATCH: 16                  # Batch size (reduce for larger backbones)
  CLIP_GRAD_NORM: 5.0                # Gradient clipping for AMP training

TEST:
  NECK_FEAT: 'before'                # Extract BN-bottleneck input (CLS token)
  FEAT_NORM: True                    # L2-normalise features before distance

LOG_ROOT: 'outputs/'                 # Checkpoint and log output directory
LOG_NAME: 'Hplus'                   # Experiment identifier tag
```

### Runtime Config Override

Any field can be overridden on the command line:

```bash
python scripts/train_dinov3.py \
  --config_file config/UrbanElementsReID_dinov3_Hplus.yml \
  SOLVER.BASE_LR 0.001 \
  SOLVER.IMS_PER_BATCH 32 \
  LOG_NAME MyExperiment
```

---

## 12. Citation & Acknowledgements

If you use this code, please cite the upstream PAT work and the Urban Elements dataset:

```bibtex
@inproceedings{ni2023part,
  title     = {Part-Aware Transformer for Generalizable Person Re-identification},
  author    = {Ni, Hao and Li, Yuke and Gao, Lianli and Shen, Heng Tao and Song, Jingkuan},
  booktitle = {ICCV},
  pages     = {11280--11289},
  year      = {2023}
}

@inproceedings{moral2024longterm,
  title     = {Long-term geo-positioned re-identification dataset of urban elements},
  author    = {Paula Moral and Alvaro García-Martín and Jose M. Martínez},
  booktitle = {IEEE ICIP},
  pages     = {124--130},
  year      = {2024},
  doi       = {10.1109/ICIP51287.2024.10647759}
}

@article{galan2025transforming,
  title   = {Transforming urban waste collection inventory: {AI}-Based container
             classification and Re-Identification},
  author  = {Javier Galán and Miguel González and Paula Moral and
             Álvaro García-Martín and José M. Martínez},
  journal = {Waste Management},
  volume  = {199},
  pages   = {25--35},
  year    = {2025},
  doi     = {10.1016/j.wasman.2025.02.051}
}
```

### Acknowledgements

- [Part-Aware Transformer (PAT)](https://github.com/liyuke65535/Part-Aware-Transformer) by liyuke65535 — the foundation of this codebase.
- [TransReID](https://github.com/damo-cv/TransReID) — the upstream ReID training framework.
- [DINOv2/v3](https://github.com/facebookresearch/dinov2) — self-supervised ViT pretraining at scale.
- [UAM Video Processing and Understanding Lab](http://www-vpu.eps.uam.es/) — challenge organisers.
