# Quick Start Guide

## Getting Started with GymPose

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/gympose.git
cd gympose
pip install -r requirements.txt
```

### 2. Prepare Your Data

#### Option A: Use Pre-annotated Data

If you already have pose keypoint annotations in CSV format, ensure they follow this structure:

```csv
class_id,class_name,image_path,kp0_x,kp0_y,kp0_conf,...,kp24_x,kp24_y,kp24_conf
```

#### Option B: Generate Annotations from Images

1. Install [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose):
   ```bash
   git clone https://github.com/JunkyByte/easy_ViTPose.git
   cd easy_ViTPose
   pip install -r requirements.txt
   ```

2. Run pose estimation on your images (refer to easy_ViTPose documentation)

3. Convert JSON annotations to CSV:
   ```bash
   python convert_json_to_csv.py --input annotations.json --output dataset.csv
   ```

### 3. Train Your First Model

#### Quick Training (XGBoost - Fastest)

For 6-class movement classification:

```bash
python average_train_xgboost_classifier.py --data path/to/your_6class_dataset.csv
```

This will:
- Train for 5 independent runs
- Save results to `results/xgboost_6class_5run_results/`
- Generate a summary file with averaged metrics
- Complete in approximately 5-10 minutes

#### Deep Learning Models

For higher accuracy with DenseNet121:

```bash
python average_train_densenet121_classifier.py --data path/to/your_6class_dataset.csv
```

Training time: ~15-20 minutes per run (75-100 minutes total for 5 runs)

### 4. View Results

After training completes, check the summary file:

```bash
cat results/xgboost_6class_5run_results/xgboost_5_run_summary.txt
```

You'll see averaged metrics across 5 runs:
```
Test Accuracy:              84.17 ± 0.18%
Top-5 Accuracy:             100.00 ± 0.00%
Precision (weighted):       83.76 ± 0.17%
Recall (weighted):          84.17 ± 0.18%
```

### 5. Evaluate on Test Data

```bash
python evaluate_model.py --model results/xgboost_6class_5run_results/run_1/model.pkl --data test_data.csv
```

## Common Workflows

### Yoga-82 Dataset Classification

1. Download Yoga-82 dataset from [official page](https://sites.google.com/view/yoga-82/home)
2. Generate pose annotations using easy_ViTPose
3. Convert to different granularities:

```bash
# Start with 82-class annotations
python convert_82_to_20_classes.py  # Creates 20-class version
python convert_20_to_6_classes.py   # Creates 6-class version
```

4. Train models on different granularities:

```bash
# 82 individual poses
python average_train_densenet121_classifier.py --data yoga_82class.csv

# 20 semantic groups
python 20class_average_train_densenet121_classifier.py --data yoga_20class.csv

# 6 broad categories
python 6class_average_train_densenet121_classifier.py --data yoga_6class.csv
```

### Custom Gymnastics Dataset

1. Organize your data with temporal phases (Phase I, II, III)
2. Generate pose annotations
3. Create both 6-class and 18-class versions
4. Train on both configurations to compare movement vs. phase-specific performance

## Tips for Best Results

### Data Quality
- Ensure good lighting and minimal occlusion in source videos
- Higher resolution images (≥640px) produce better pose estimates
- Multi-view capture reduces self-occlusion

### Model Selection
- **XGBoost**: Fast training, good for quick experiments (5-10 min)
- **DenseNet121**: Best overall accuracy (75-100 min)
- **FT-Transformer**: Good balance of speed and accuracy (60-80 min)
- **ResNeXt50**: Moderate accuracy, moderate training time (40-60 min)

### Training Strategy
1. Start with XGBoost to validate data quality
2. If XGBoost performs well (>80%), try DenseNet121
3. Use 5-run averaging for robust performance estimates
4. Monitor validation accuracy to detect overfitting

## Troubleshooting

### Low Accuracy (<70%)
- Check class distribution (imbalanced?)
- Verify keypoint quality (confidence scores)
- Ensure correct CSV format
- Try data augmentation or class balancing

### Out of Memory
- Reduce batch size in training scripts
- Use smaller model (try XGBoost first)
- Process on CPU if GPU memory is limited

### Slow Training
- Start with XGBoost baseline
- Reduce number of runs from 5 to 3
- Use GPU acceleration for neural networks

## Next Steps

- Check [TCF_DATASET_PREPROCESSING.md](TCF_DATASET_PREPROCESSING.md) for dataset preparation pipeline
- See [KEYPOINTS.md](KEYPOINTS.md) for detailed keypoint structure
- Review [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for repository organization

## Need Help?

Open an issue on GitHub with:
- Description of your problem
- Sample of your CSV data (first 5 rows)
- Error messages or unexpected output
- System information (Python version, GPU/CPU)
