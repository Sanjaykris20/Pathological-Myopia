# Explainable Multi-Task Deep Learning for Pathological Myopia Analysis

A research-oriented deep learning pipeline for **pathological myopia
(PM) analysis from color fundus photographs**, combining multi-dataset
harmonization, preprocessing, six-model benchmarking, controlled
optimization, multi-task learning, quantitative phenotyping, and
explainability.

> **Research implementation — not a clinically validated diagnostic
> system.**

## Overview

Pathological myopia is associated with progressive structural changes in
the posterior segment of the eye. Fundus photographs can reveal
abnormalities such as tessellated fundus, atrophy, lacquer cracks, and
Fuchs spots.

This project goes beyond binary PM classification by investigating:

- PM classification
- Five-level severity assessment
- Fovea localization
- Retinal structure/lesion segmentation
- Mask-based quantitative phenotyping
- Spatial analysis
- Grad-CAM explainability

### Workflow

``` text
Public Fundus Datasets
        ↓
Dataset Harmonization
        ↓
Quality Control + Duplicate Handling
        ↓
Group-Aware Train / Validation / Test Split
        ↓
Retinal Preprocessing
  ├── FOV detection / crop
  ├── 224×224 resize
  ├── LAB conversion
  ├── CLAHE
  └── normalization
        ↓
Six-Model PM Classification Benchmark
  ├── ResNet-50
  ├── DenseNet-121
  ├── EfficientNet-B4
  ├── ViT-B/16
  ├── Swin-Tiny
  └── RETFound
        ↓
Validation-Based Benchmark Selection
        ↓
Controlled Optimization of Remaining 5 Models
        ↓
Swin-Tiny Final Backbone
        ↓
Multi-Task Learning
  ├── PM Classification
  ├── Severity Classification
  ├── Fovea Localization
  └── Segmentation
        ↓
Quantitative Phenotyping
        ↓
Explainability + Final Evaluation
```

## Research Objectives

### 1. Pathological Myopia Classification

Binary prediction:

``` text
0 → Non-PM
1 → PM
```

### 2. Severity Classification

Five levels:

| Class | Description        |
|------:|--------------------|
|     0 | No lesion          |
|     1 | Tessellated fundus |
|     2 | Diffuse atrophy    |
|     3 | Patchy atrophy     |
|     4 | Macular atrophy    |

### 3. Fovea Localization

Prediction of normalized `(x, y)` foveal coordinates.

### 4. Segmentation

The final experiment activates segmentation heads only when sufficient
annotations are available. The active final heads are:

- Optic disc
- Lacquer cracks
- Fuchs spots

## Key Contributions

- Harmonization of complementary public fundus datasets.
- Annotation-aware training: **missing annotation ≠ negative
  annotation**.
- Benchmarking of CNN, Transformer, and retinal foundation-model
  architectures.
- Controlled optimization of the five non-benchmark models.
- Multi-task learning using a shared Swin-Tiny representation.
- Mask-based quantitative structural phenotyping.
- Grad-CAM and mask-aware activation analysis.

## Datasets

| Dataset     |    Images | Main contribution                      |
|-------------|----------:|----------------------------------------|
| PALM        |     1,200 | PM classification + fovea localization |
| MMAC 2023   |     1,572 | Severity + lesion annotations          |
| HRF-Seg+    |        45 | Optic-disc/structural segmentation     |
| MCTN Fundus |       289 | Additional fundus images               |
| **Total**   | **3,106** | Combined source pool                   |

Different tasks use different subsets of the available annotations.

### PALM

PM/non-PM information and foveal localization.

### MMAC 2023

Five-level severity and lesion annotations including lacquer cracks,
CNV, Fuchs spots, atrophy, and tessellation.

### HRF-Seg+

High-resolution structural segmentation information, particularly useful
for optic-disc analysis.

### MCTN Fundus

Additional fundus-image diversity; it is not a primary source of PM
classification labels in the current training pipeline.

## Data Harmonization and Leakage Control

A common metadata representation is created for heterogeneous datasets.

Typical fields include:

``` text
dataset
source_id
image_path
pm_label
severity
fovea_x
fovea_y
mask_disc
mask_atrophy
mask_lesion_lc
mask_lesion_cnv
mask_lesion_fs
mask_tessellation
```

Unavailable annotations remain unknown.

The split is group-aware so that related images are kept within the same
train/validation/test partition. Exact duplicates are identified using
file hashes, and near-duplicate analysis is used as an additional
safeguard.

> Patient-level IDs are not consistently available across all source
> datasets, so complete patient-level separation cannot always be
> guaranteed.

## Preprocessing

``` text
Fundus Image
    ↓
Retinal/FOV Detection
    ↓
Background Crop
    ↓
224×224
    ↓
LAB
    ↓
CLAHE on L channel
    ↓
Normalization
    ↓
Model Input
```

The same geometric transformation is applied to relevant fovea
coordinates and segmentation masks. A fundus-region mask is retained for
quantitative analysis.

## Six-Model Benchmark

| Model           | Family                   | Parameters |
|-----------------|--------------------------|-----------:|
| ResNet-50       | CNN                      |      23.5M |
| DenseNet-121    | CNN                      |       7.0M |
| EfficientNet-B4 | CNN                      |      17.6M |
| ViT-B/16        | Transformer              |      85.8M |
| Swin-Tiny       | Hierarchical Transformer |      27.5M |
| RETFound        | Retinal foundation model |     303.3M |

### Architectural motivation

- **ResNet-50:** residual/skip connections; strong classical CNN
  baseline.
- **DenseNet-121:** dense feature reuse and strong gradient flow.
- **EfficientNet-B4:** MBConv, squeeze-and-excitation, and efficient
  scaling.
- **ViT-B/16:** patch-based global self-attention.
- **Swin-Tiny:** hierarchical shifted-window attention.
- **RETFound:** retinal-domain pretrained representation.

## Baseline Training

``` text
Input size        : 224×224
Batch size        : 16
Optimizer         : AdamW
Learning rate     : 1e-4
Weight decay      : 1e-2
Maximum epochs    : 15
Early stopping    : patience 4
Gradient clipping : 1.0
Scheduler         : cosine
Loss              : BCEWithLogitsLoss
Mixed precision   : CUDA-dependent
```

Validation performance is used for model selection; the test set is
reserved for final reporting.

## Baseline Results

| Model           |  Val AUROC |  Val AUPRC | Test AUROC | Test AUPRC |    Test F1 |
|-----------------|-----------:|-----------:|-----------:|-----------:|-----------:|
| ResNet-50       |     0.9818 |     0.9808 |     0.9921 |     0.9885 |     0.9000 |
| DenseNet-121    |     0.9867 |     0.9848 |     0.9959 |     0.9942 |     0.9354 |
| EfficientNet-B4 |     0.9855 |     0.9833 |     0.9880 |     0.9827 |     0.9009 |
| ViT-B/16        |     0.9876 |     0.9863 |     0.9957 |     0.9935 |     0.9383 |
| **Swin-Tiny**   | **0.9910** | **0.9895** | **0.9968** | **0.9954** | **0.9412** |
| RETFound        |     0.9894 |     0.9899 |     0.9958 |     0.9936 |     0.9554 |

### Benchmark

**Swin-Tiny** was selected because it achieved the highest validation
AUROC:

> **Validation AUROC = 0.9910**

Validation AUROC is the primary selection criterion, with AUPRC as the
tie-breaker. Test performance was not used for selection.

## Controlled Optimization

The remaining five models were tested with targeted changes including:

- Head learning-rate ×10
- Positive-class weighting

| Model           | Experiment      | Val AUROC | Δ Val AUROC |
|-----------------|-----------------|----------:|------------:|
| ResNet-50       | Head LR ×10     |    0.9882 | **+0.0064** |
| ResNet-50       | Positive weight |    0.9860 |     +0.0043 |
| DenseNet-121    | Head LR ×10     |    0.9876 |     +0.0009 |
| DenseNet-121    | Positive weight |    0.9908 |     +0.0041 |
| EfficientNet-B4 | Head LR ×10     |    0.9849 |     −0.0006 |
| EfficientNet-B4 | Positive weight |    0.9847 |     −0.0008 |
| ViT-B/16        | Head LR ×10     |    0.9896 |     +0.0020 |
| ViT-B/16        | Positive weight |    0.9855 |     −0.0021 |
| RETFound        | Head LR ×10     |    0.9895 |     +0.0001 |
| RETFound        | Positive weight |    0.9901 |     +0.0007 |

A predefined **+0.005 validation-AUROC acceptance margin** was used.
Only ResNet-50 + Head LR ×10 crossed the margin. No optimized model
surpassed the Swin-Tiny benchmark.

## Final Multi-Task Architecture

``` text
Fundus Image
     ↓
Shared Swin-Tiny Encoder
     ↓
Shared Feature Representation
     ├── PM Classification
     ├── Severity Classification
     ├── Fovea Localization
     └── Segmentation Decoder
           ├── Optic Disc
           ├── Lacquer Cracks
           └── Fuchs Spots
```

The shared encoder allows multiple related tasks to use a common retinal
representation.

## Final Results

### PM Classification

| Metric      |     Result |
|-------------|-----------:|
| AUROC       | **0.9962** |
| AUPRC       | **0.9942** |
| F1          | **0.9474** |
| Precision   |     0.9053 |
| Sensitivity |     0.9935 |
| Specificity |     0.9339 |
| ECE         |     0.0362 |

### Severity

| Metric   |     Result |
|----------|-----------:|
| Accuracy | **0.8900** |
| Macro-F1 |     0.7250 |
| QWK      | **0.9432** |

### Fovea Localization

| Metric       |      Result |
|--------------|------------:|
| Mean error   | **9.08 px** |
| Median error |     6.70 px |
| ≤10 px       |      70.59% |
| ≤20 px       |  **93.58%** |
| ≤40 px       |      98.40% |

### Segmentation

| Head           |       Dice |        IoU |
|----------------|-----------:|-----------:|
| Optic Disc     | **1.0000** | **1.0000** |
| Lacquer Cracks |     0.1362 |     0.0801 |
| Fuchs Spots    |     0.0934 |     0.0551 |

Lesion segmentation results should be interpreted cautiously because the
corresponding test sets are small.

## Quantitative Phenotyping

Predicted segmentation masks are used for post-processing measurements
such as:

- Lesion area in pixels
- Percentage of fundus area
- Lesion count
- Spatial distribution
- Fovea-relative distances
- Optic-disc-relative measurements
- Severity/lesion relationships

No physical `mm²` values are claimed because a reliable
pixel-to-millimeter calibration is unavailable.

## Explainability

Grad-CAM is applied to investigate regions contributing to PM
predictions.

Mask-aware activation enrichment:

| Structure      | Enrichment |
|----------------|-----------:|
| Optic Disc     |      1.00× |
| Lacquer Cracks |  **4.98×** |
| Fuchs Spots    |  **2.14×** |

These are exploratory attribution analyses, not clinical causal
evidence.

## Evaluation Metrics

### Classification

- AUROC
- AUPRC
- Accuracy
- Precision
- Recall/Sensitivity
- Specificity
- F1
- ECE

### Severity

- Accuracy
- Macro-F1
- Quadratic Weighted Kappa (QWK)

### Fovea

- Mean/median Euclidean error
- Success within 10/20/40 pixels

### Segmentation

- Dice coefficient
- IoU

## Inference

A new fundus image follows the same preprocessing pipeline:

``` text
New Fundus Image
       ↓
Preprocessing
       ↓
Swin-Tiny
       ↓
PM Probability
Severity
Fovea Coordinates
Segmentation Masks
       ↓
Quantitative / Visualization Outputs
```

## Project Structure

``` text
pathological-myopia/
│
├── pathological-myopia.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── PALM/
│   ├── MMAC2023/
│   ├── HRF-Seg+/
│   └── MCTN/
│
├── experiments/
│   ├── preprocessing/
│   ├── checkpoints/
│   ├── metrics/
│   ├── inference/
│   └── explainability/
│
└── results/
```

Dataset files and pretrained checkpoints should not be committed unless
their licenses permit redistribution.

## Installation

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

python -m venv venv
```

Windows:

``` bash
venv\Scriptsctivate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Main libraries:

``` text
Python
PyTorch
TorchVision
timm
OpenCV
Albumentations
NumPy
Pandas
SciPy
scikit-learn
Pillow
Matplotlib
tqdm
```

## Running the Notebook

``` bash
jupyter notebook
```

Open:

``` text
pathological-myopia.ipynb
```

Recommended order:

1.  Configure dataset paths.
2.  Verify datasets and pretrained checkpoints.
3.  Build dataset metadata.
4.  Run preprocessing.
5.  Generate train/validation/test splits.
6.  Train six baseline models.
7.  Evaluate baselines.
8.  Select Swin-Tiny.
9.  Run controlled optimization on the remaining five models.
10. Train/evaluate the multi-task Swin-Tiny model.
11. Generate quantitative phenotyping.
12. Generate explainability outputs.

## Reproducibility

A fixed seed is used:

``` python
SEED = 42
```

The notebook records experiment configuration, dataset statistics, model
settings, and evaluation outputs.

Exact reproducibility can still depend on GPU, CUDA, PyTorch versions,
and deterministic execution settings.

## Limitations

Important current limitations include:

- Patient-level identifiers are not consistently available across all
  datasets.
- Near-duplicate handling cannot guarantee perfect patient-level
  separation.
- Dataset/camera distribution may introduce dataset-specific shortcuts.
- Some segmentation test sets are very small.
- Small lesions may be difficult to represent at 224×224.
- A common baseline recipe may not be optimal for every architecture.
- RETFound is much larger than the other models and may require
  specialized fine-tuning.
- The optimization stage is limited to selected controlled experiments.
- The segmentation decoder is relatively lightweight.
- A dedicated multi-task ablation study is needed to prove that joint
  learning improves each task.
- External validation on independent datasets is still required.
- Quantitative lesion measurements require stronger validation.
- Grad-CAM indicates attribution, not clinical causality.

## Research Status

This repository is intended for **academic and research purposes** and
is not a clinically validated diagnostic system.

Further work would require:

- External multi-center validation
- Strong patient-level evaluation
- Prospective testing
- Larger lesion cohorts
- Calibration validation
- Clinical expert review
- Clinical/regulatory validation

## Base Paper

**Hemelings R, Elen B, Blaschko MB, Jacob J, Stalmans I, De Boever
P.**  
*Pathological myopia classification with simultaneous lesion
segmentation using deep learning.*  
Computer Methods and Programs in Biomedicine. 2021;199:105920.  
DOI: `10.1016/j.cmpb.2020.105920`

The base paper motivates the combination of PM classification and lesion
analysis; this project extends that direction through multi-model
benchmarking, controlled optimization, multi-task analysis, quantitative
phenotyping, and explainability.

## Citation

``` bibtex
@article{hemelings2021pathological,
  title={Pathological myopia classification with simultaneous lesion segmentation using deep learning},
  author={Hemelings, R and Elen, B and Blaschko, MB and Jacob, J and Stalmans, I and De Boever, P},
  journal={Computer Methods and Programs in Biomedicine},
  volume={199},
  pages={105920},
  year={2021},
  doi={10.1016/j.cmpb.2020.105920}
}
```

Please also cite the original datasets and pretrained models according
to their respective licenses and publications.

## Disclaimer

This project is for **academic and research purposes only**.

Model predictions are not medical diagnoses and are not intended to
replace evaluation by a qualified ophthalmologist.
