# GymPose Repository Structure

## Overview

This repository contains all scripts and documentation for the GymPose project: A New Benchmark and Pose-based Framework for Fine-grained Action Classification.

## Repository Contents

### Directory Structure

```
gympose_repo/
├── scripts/              # Training scripts
│   ├── train_yoga82_*.py (Yoga-82 / GymPose 18-class)
│   ├── train_yoga20_*.py (Yoga-20 semantic groups)
│   └── train_yoga6_*.py  (Yoga-6 broad categories)
├── preprocessing/        # Data processing scripts
│   ├── convert_*.py      (Dataset conversion)
│   ├── prepare_*.py      (Data preparation)
│   └── evaluate_model.py (Model evaluation)
├── docs/                 # Documentation
│   ├── KEYPOINTS.md
│   ├── QUICKSTART.md
│   ├── TCF_DATASET_PREPROCESSING.md
│   └── REPOSITORY_STRUCTURE.md
├── README.md             # Main documentation
├── requirements.txt      # Dependencies
└── .gitignore           # Git ignore rules
```

### Training Scripts (`scripts/`)

#### Yoga-82 / GymPose Training Scripts
These scripts work with both Yoga-82 (82 poses) and GymPose (18-class phase-specific):
- `train_yoga82_densenet121_classifier.py` - DenseNet121 model
- `train_yoga82_resnext50_classifier.py` - ResNeXt50 model
- `train_yoga82_fttransformer_classifier.py` - FT-Transformer model
- `train_yoga82_xgboost_classifier.py` - XGBoost model
- `train_yoga82_efficientnetv2s_classifier.py` - EfficientNetV2-S model

#### Yoga-20 Training Scripts
For Yoga-82 dataset with 20 semantic groups:
- `train_yoga20_densenet121_classifier.py`
- `train_yoga20_resnext50_classifier.py`
- `train_yoga20_fttransformer_classifier.py`
- `train_yoga20_xgboost_classifier.py`
- `train_yoga20_efficientnetv2s_classifier.py`

#### Yoga-6 Training Scripts
For Yoga-82 dataset with 6 broad categories:
- `train_yoga6_densenet121_classifier.py`
- `train_yoga6_resnext50_classifier.py`
- `train_yoga6_fttransformer_classifier.py`
- `train_yoga6_xgboost_classifier.py`

### Data Processing Scripts

#### Conversion Scripts
- `convert_json_to_csv.py` - Convert ViTPose JSON annotations to CSV format
- `convert_82_to_20_classes.py` - Convert 82-class to 20-class granularity
- `convert_20_to_6_classes.py` - Convert 20-class to 6-class granularity
- `convert_rtmpose_82_to_20.py` - Convert RTMPose 82-class to 20-class
- `convert_rtmpose_20_to_6.py` - Convert RTMPose 20-class to 6-class

#### Preparation Scripts
- `prepare_18class_training_csv.py` - Prepare 18-class CSV for phase-specific training

### Evaluation Scripts
- `evaluate_model.py` - Evaluate trained models on test data

### Documentation Files (`docs/`)

- `KEYPOINTS.md` - Detailed keypoint format and body part mapping
- `QUICKSTART.md` - Quick start guide for new users
- `TCF_DATASET_PREPROCESSING.md` - GymPose dataset preprocessing pipeline
- `REPOSITORY_STRUCTURE.md` - This file - repository organization guide

### Configuration Files

- `requirements.txt` - Python package dependencies
- `.gitignore` - Git ignore rules for data files, results, and temporary files

## Key Features

### Pose Estimation Integration
- Uses [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) for pose estimation
- Supports both ViTPose-H and RTMPose annotations
- COCO-WholeBody format (25 keypoints for ViTPose, 26 for RTMPose)

### Keypoint Structure
Each annotation follows this format:
```
class_id, class_name, image_path, 
kp0_x, kp0_y, kp0_conf, ..., kp24_x, kp24_y, kp24_conf
```

For 18-class configurations, additional columns:
```
..., phase_name, phase_id, combined_class_id
```

### Classification Granularities

1. **6-Class**: Broad movement categories
2. **18-Class**: Phase-specific movements (6 movements × 3 phases)
3. **20-Class**: Semantic pose groups (Yoga-82)
4. **82-Class**: Individual poses (Yoga-82)

### Model Architectures

- **DenseNet121**: Best accuracy, convolutional approach
- **ResNeXt50**: Cardinality-based residual network
- **FT-Transformer**: Transformer for tabular data
- **XGBoost**: Gradient boosting baseline
- **EfficientNetV2-S**: Efficient scaling (experimental)

## Excluded from Repository

The following are NOT included and should NOT be committed:

### Data Files
- Raw images (*.jpg, *.png, *.jpeg)
- Raw videos (*.mp4, *.avi, *.mov)
- CSV datasets (*.csv)
- JSON annotations (*.json)
- Preprocessed data files

### Model Files
- Trained model checkpoints (*.pth, *.pt, *.pkl)
- Training logs and results (results/ directory)

### External Dependencies
- easy_ViTPose repository
- YOLO models
- Pre-trained weights

### Dataset Sources
- **Yoga-82**: Download from [official page](https://sites.google.com/view/yoga-82/home)
- **GymPose**: Contact authors for access

## Usage Examples

### Train 6-class model
```bash
python average_train_xgboost_classifier.py --data path/to/6class_dataset.csv
```

### Train 18-class phase-specific model
```bash
python average_train_densenet121_classifier.py --data path/to/18class_dataset.csv
```

### Convert dataset granularity
```bash
python convert_82_to_20_classes.py
python convert_20_to_6_classes.py
```

### Evaluate model
```bash
python evaluate_model.py --model path/to/model.pth --data test_data.csv
```

## Expected Outputs

After training, results are saved in `results/` directory:
```
results/
├── model_6class_5run_results/
│   ├── run_1/
│   │   ├── model.pth (or .pkl for XGBoost)
│   │   ├── training_history.csv
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   ├── run_2/ ... run_5/
│   └── model_5_run_summary.txt
```

Summary file contains:
- Test Accuracy (mean ± std)
- Top-5 Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

## Performance Benchmarks

### GymPose Dataset (ViTPose annotations)

| Model | TCF-6 Accuracy | TCF-18 Accuracy |
|-------|----------------|-----------------|
| DenseNet121 | 95.15 ± 0.57% | 90.04 ± 0.45% |
| FT-Transformer | 94.83 ± 0.13% | 89.07 ± 0.69% |
| XGBoost | 93.75 ± 0.65% | 88.13 ± 0.64% |
| ResNeXt50 | 91.50 ± 0.53% | 85.81 ± 1.27% |

For detailed preprocessing pipeline, see [TCF_DATASET_PREPROCESSING.md](TCF_DATASET_PREPROCESSING.md).

## Acknowledgments

- **[easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)** - Simplified ViTPose implementation for pose estimation
- **[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)** - Real-time multi-person pose estimation framework
- **[Yoga-82 Dataset](https://sites.google.com/view/yoga-82/home)** - Yoga pose dataset for classification benchmarks
