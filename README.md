# GymPose: A New Benchmark and Pose-based Framework for Fine-grained Action Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A pose-based framework for fine-grained classification of artistic gymnastics movements using keypoint annotations. This repository provides training scripts, preprocessing pipelines, and evaluation tools for movement recognition and temporal phase detection.

## Overview

This project implements a hierarchical classification approach for gymnastics movements using human pose keypoints extracted from video frames. The framework supports:

- **Dual-granularity classification**: 6-class movement recognition and 18-class phase-specific classification
- **Multiple pose estimators**: ViTPose and RTMPose annotations
- **Four classification architectures**: DenseNet121, ResNeXt50, FT-Transformer, and XGBoost
- **Multi-view support**: Compatible with single-view and multi-camera setups

## Dataset

### GymPose Dataset

The GymPose dataset contains 16,782 annotated frames of six artistic gymnastics movements:
- TCF Flik Flak
- TCF Geriye Toplu Salto (backward tucked somersault)
- TCF Geyik Sicrama (deer jump)
- TCF Makasli Geyik (scissor deer)
- TCF Makasli Geyik Yarim (half scissor deer)
- TCF Spagat (split)

Each movement is divided into three temporal phases:
- **Phase I**: Starting position
- **Phase II**: Mid-movement
- **Phase III**: Ending position

### Yoga-82 Dataset

The framework also supports the Yoga-82 dataset for pose-based yoga classification. Download the original dataset from the [Yoga-82 Project Page](https://sites.google.com/view/yoga-82/home).

## Pose Estimation

This project uses pose keypoint annotations following the COCO-WholeBody format. Pose estimation can be performed using ViTPose or RTMPose frameworks.

### ViTPose

[easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) provides a simplified implementation of ViTPose for human pose estimation. ViTPose-H extracts 25 body keypoints per person with high accuracy.

### RTMPose

[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) is a real-time multi-person pose estimation framework from OpenMMLab. RTMPose extends the keypoint format to 26 keypoints, offering real-time inference capabilities.

### Keypoint Format

Each annotation contains 25 keypoints (ViTPose) or 26 keypoints (RTMPose) with the following structure. For detailed keypoint indices and body part mappings, see [docs/KEYPOINTS.md](docs/KEYPOINTS.md).

Each keypoint is represented by three values: `(x, y, confidence)`, where `(x, y)` are pixel coordinates and `confidence` is a score in the range [0, 1].

### CSV Data Format

Training scripts expect CSV files with the following structure:
```
class_id, class_name, image_path, kp0_x, kp0_y, kp0_conf, kp1_x, kp1_y, kp1_conf, ..., kp24_x, kp24_y, kp24_conf
```

For 18-class (phase-specific) configurations, additional columns are included:
```
..., phase_name, phase_id, combined_class_id
```

The classification models use only the `(x, y)` coordinate pairs, resulting in 50-dimensional feature vectors (25 keypoints × 2 coordinates).

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch 1.12+
- torchvision
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- tqdm

### Pose Estimation Setup

#### ViTPose

For ViTPose-based pose estimation, install [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose):

```bash
git clone https://github.com/JunkyByte/easy_ViTPose.git
cd easy_ViTPose
pip install -r requirements.txt
```

Refer to the easy_ViTPose repository for detailed installation and usage instructions.

#### RTMPose

For RTMPose-based pose estimation, install [MMPose](https://github.com/open-mmlab/mmpose) with RTMPose support:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose>=1.0.0"
```

Refer to the [RTMPose documentation](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) for detailed setup and usage.

## Usage

### Training Models

All training scripts are located in the `scripts/` directory and support 5-run averaging with stratified cross-validation. Models are trained with early stopping based on validation accuracy.

#### GymPose Dataset (6-class and 18-class)

For the GymPose gymnastics dataset, use the Yoga-82 training scripts with your GymPose CSV files:

```bash
# 6-class movement classification
python scripts/train_yoga82_xgboost_classifier.py --data path/to/gympose_6class.csv
python scripts/train_yoga82_densenet121_classifier.py --data path/to/gympose_6class.csv

# 18-class phase-specific classification
python scripts/train_yoga82_densenet121_classifier.py --data path/to/gympose_18class.csv
```

#### Yoga-82 Dataset

Train on the Yoga-82 dataset with different granularity levels:

**Yoga-82 (82 individual poses):**
```bash
python scripts/train_yoga82_xgboost_classifier.py --data path/to/yoga_82class.csv
python scripts/train_yoga82_densenet121_classifier.py --data path/to/yoga_82class.csv
python scripts/train_yoga82_resnext50_classifier.py --data path/to/yoga_82class.csv
python scripts/train_yoga82_fttransformer_classifier.py --data path/to/yoga_82class.csv
```

**Yoga-20 (20 semantic groups):**
```bash
python scripts/train_yoga20_xgboost_classifier.py --data path/to/yoga_20class.csv
python scripts/train_yoga20_densenet121_classifier.py --data path/to/yoga_20class.csv
python scripts/train_yoga20_resnext50_classifier.py --data path/to/yoga_20class.csv
python scripts/train_yoga20_fttransformer_classifier.py --data path/to/yoga_20class.csv
```

**Yoga-6 (6 broad categories):**
```bash
python scripts/train_yoga6_xgboost_classifier.py --data path/to/yoga_6class.csv
python scripts/train_yoga6_densenet121_classifier.py --data path/to/yoga_6class.csv
python scripts/train_yoga6_resnext50_classifier.py --data path/to/yoga_6class.csv
python scripts/train_yoga6_fttransformer_classifier.py --data path/to/yoga_6class.csv
```

### Output Structure

Training results are saved in the `results/` directory:

```
results/
├── model_6class_5run_results/
│   ├── run_1/
│   │   ├── model.pth
│   │   ├── training_history.csv
│   │   └── confusion_matrix.png
│   ├── run_2/
│   ├── ...
│   ├── run_5/
│   └── model_5_run_summary.txt
└── model_18class_5run_results/
    └── ...
```

Summary files contain averaged metrics across 5 runs:
- Test Accuracy (mean ± std)
- Top-5 Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

## Preprocessing Scripts

All preprocessing scripts are located in the `preprocessing/` directory.

### Dataset Conversion (Yoga-82)

Convert between different class granularities for the Yoga-82 dataset:

```bash
# Convert 82-class to 20-class
python preprocessing/convert_82_to_20_classes.py

# Convert 20-class to 6-class
python preprocessing/convert_20_to_6_classes.py
```

### ViTPose Annotation Processing

Process raw ViTPose JSON annotations to CSV format:

```bash
# Convert JSON annotations to CSV
python preprocessing/convert_json_to_csv.py --input annotations.json --output dataset.csv
```

### RTMPose Annotation Processing

Convert RTMPose annotations to different granularities:

```bash
# Convert RTMPose 82-class to 20-class (Yoga-82)
python preprocessing/convert_rtmpose_82_to_20.py

# Convert RTMPose 20-class to 6-class (Yoga-82)
python preprocessing/convert_rtmpose_20_to_6.py
```

### GymPose Dataset Preparation

Prepare 18-class CSV for phase-specific training:

```bash
# Prepare GymPose 18-class training CSV
python preprocessing/prepare_18class_training_csv.py
```

This script converts the `combined_class_id` column to `class_id` for phase-specific classification.

## Evaluation

Evaluate trained models on test sets:

```bash
python preprocessing/evaluate_model.py --model path/to/model.pth --data path/to/test_data.csv
```

## Results

Performance metrics for the GymPose dataset (ViTPose annotations):

| Model | TCF-6 Accuracy | TCF-18 Accuracy |
|-------|----------------|-----------------|
| DenseNet121 | 95.15 ± 0.57% | 90.04 ± 0.45% |
| ResNeXt50 | 91.50 ± 0.53% | 85.81 ± 1.27% |
| FT-Transformer | 94.83 ± 0.13% | 89.07 ± 0.69% |
| XGBoost | 93.75 ± 0.65% | 88.13 ± 0.64% |

For detailed preprocessing pipeline, see [TCF Dataset Preprocessing](docs/TCF_DATASET_PREPROCESSING.md).

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started quickly with training
- **[Keypoint Structure](docs/KEYPOINTS.md)** - Detailed keypoint format and indices
- **[TCF Dataset Preprocessing](docs/TCF_DATASET_PREPROCESSING.md)** - GymPose preprocessing pipeline
- **[Repository Structure](docs/REPOSITORY_STRUCTURE.md)** - Repository organization

## Acknowledgments

- **[easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)** - Simplified ViTPose implementation for pose estimation
- **[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)** - Real-time multi-person pose estimation framework
- **[Yoga-82 Dataset](https://sites.google.com/view/yoga-82/home)** - Yoga pose dataset for classification benchmarks
- **ViTPose and RTMPose teams** - Original pose estimation architectures

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
