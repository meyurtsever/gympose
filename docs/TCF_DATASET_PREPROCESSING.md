# TCF Gymnastics Dataset Preprocessing

## Model Configuration

### Detection Model
| Parameter | Value |
|-----------|-------|
| Model | YOLO11x |
| Weights | yolo11x.pt |
| Parameters | 68.2M |
| mAP | 53.0 |
| Input Size | 640px |
| Detection Class | human |
| Confidence Threshold | 0.4 |

### Pose Estimation Model
| Parameter | Value |
|-----------|-------|
| Model | ViTPose-H |
| Weights | vitpose-h-coco_25.pth |
| Backbone | ViT-Huge |
| AP (COCO) | 79.1 |
| Input Size | 256×192 |
| Keypoints | 25 (COCO-WholeBody) |

## Dataset Structure

### Source Dataset
The TCF_DATASET contains 6 gymnastics movement classes. Each class is organized with the following hierarchy:
- Class folder (e.g., "TCF Makaslı Geyik Yarım Donuslu")
  - Individual folders (e.g., "1.sporcu")
    - Attempt folders (e.g., "1.Acı")
      - Phase folders (e.g., "1.Bolum", "2.Bolum", "3.Bolum")
        - Image files

### Preprocessing Steps

#### Step 1: Dataset Flattening
The repeat folder structure was flattened. Phase folders with multiple attempts were merged into single phase folders. The script `preprocess_tcf_dataset.py` was executed.

#### Step 2: Path Length Resolution
The class "TCF Makaslı Geyik Yarım Donuslu" failed processing due to path length exceeding Windows limits (207 characters). The folder was renamed to "TCF_Makasli_Geyik_Yarim". Turkish characters were converted to ASCII equivalents. Path length was reduced from 207 to 143 characters (64 character reduction).

#### Step 3: Pose Estimation
The script `process_gymnastics_dataset.py` was executed. YOLO11x detected persons in images. ViTPose-H extracted 25 keypoints for each detected person. Detection scores were recorded for each person. Phase information was preserved from folder structure.

#### Step 4: Metadata Correction
Phase information extraction was corrected. The class name for "TCF_Makasli_Geyik_Yarim" was fixed from individual folder names to the correct class name. Phase typos were identified and corrected:
- "1.Bölüm" → "1.Bolum" (45 images)
- "2.Bölüm" → "2.Bolum" (30 images)
- "3.Bölüm" → "3.Bolum" (35 images)
- "3..Bolum" → "3.Bolum" (4 images)

Total corrections: 114 images in TCF Geyik Sicrama class.

#### Step 5: Dataset Consolidation
Six class-specific JSON files were merged into `gymnastics_complete_dataset.json`. The file contains 16,791 annotations with keypoints, detection scores, and phase information.

#### Step 6: Quality Analysis
Zero-detection analysis was performed. 9 images (0.05%) contained no detected persons. 16,782 images (99.95%) contained at least one detected person.

Zero detections by class:
- TCF Makasli Geyik: 6 images (0.09%)
- TCF Geriye Toplu Salto: 1 image (0.08%)
- TCF Geyik Sicrama: 1 image (0.07%)
- TCF Spagat: 1 image (0.06%)

Zero detections by phase:
- 2.Bolum: 6 images (0.13%)
- 3.Bolum: 2 images (0.03%)
- 1.Bolum: 1 image (0.02%)

#### Step 7: Person Selection
The script `preprocess_gymnastics_dataset.py` was executed. Images with zero detections were excluded (9 images). Images with multiple persons were processed. The person with the highest detection score was selected for each image. 7,629 images (45.45%) contained multiple persons. 9,153 images (54.50%) contained a single person. All annotations were normalized to contain a single person with ID "0".

#### Step 8: CSV Generation
The script `create_tcf_csvs.py` was executed. Two CSV files were generated:
1. `tcf_poses_dataset_vit_6class.csv` - 6 classes
2. `tcf_poses_dataset_vit_18class.csv` - 18 classes (6 movements × 3 phases)

CSV format:
- Columns: class_id, class_name, image_path, 25 keypoints (x, y, confidence)
- Keypoint format: COCO-WholeBody (25 body keypoints)

## Dataset Statistics

### Initial Dataset (After Pose Estimation)
| Metric | Value |
|--------|-------|
| Total Annotations | 16,791 |
| Classes | 6 |
| Unique Phases | 3 |
| Detection Success Rate | 99.95% |

### Class Distribution (Initial)
| Class ID | Class Name | Samples |
|----------|-----------|---------|
| 0 | TCF Flik Flak | 1,170 |
| 1 | TCF Geriye Toplu Salto | 1,187 |
| 2 | TCF Geyik Sicrama | 1,497 |
| 3 | TCF Makasli Geyik | 6,753 |
| 4 | TCF Makasli Geyik Yarim | 4,368 |
| 5 | TCF Spagat | 1,816 |
| **Total** | | **16,791** |

### Final Dataset (After Preprocessing)
| Metric | Value |
|--------|-------|
| Total Annotations | 16,782 |
| Classes (6-class) | 6 |
| Classes (18-class) | 18 |
| Excluded (Zero Detection) | 9 |
| Multiple Persons Processed | 7,629 |
| Single Person Kept | 9,153 |

### Class Distribution (Final - 6 Classes)
| Class ID | Class Name | Samples |
|----------|-----------|---------|
| 0 | TCF Flik Flak | 1,170 |
| 1 | TCF Geriye Toplu Salto | 1,186 |
| 2 | TCF Geyik Sicrama | 1,496 |
| 3 | TCF Makasli Geyik | 6,747 |
| 4 | TCF Makasli Geyik Yarim | 4,368 |
| 5 | TCF Spagat | 1,815 |
| **Total** | | **16,782** |

### Class Distribution (Final - 18 Classes)
| Class ID | Class Name | Samples |
|----------|-----------|---------|
| 0 | TCF Flik Flak_1.Bolum | 444 |
| 1 | TCF Flik Flak_2.Bolum | 325 |
| 2 | TCF Flik Flak_3.Bolum | 401 |
| 3 | TCF Geriye Toplu Salto_1.Bolum | 437 |
| 4 | TCF Geriye Toplu Salto_2.Bolum | 415 |
| 5 | TCF Geriye Toplu Salto_3.Bolum | 334 |
| 6 | TCF Geyik Sicrama_1.Bolum | 545 |
| 7 | TCF Geyik Sicrama_2.Bolum | 449 |
| 8 | TCF Geyik Sicrama_3.Bolum | 502 |
| 9 | TCF Makasli Geyik_1.Bolum | 2,579 |
| 10 | TCF Makasli Geyik_2.Bolum | 2,053 |
| 11 | TCF Makasli Geyik_3.Bolum | 2,115 |
| 12 | TCF Makasli Geyik Yarim_1.Bolum | 1,542 |
| 13 | TCF Makasli Geyik Yarim_2.Bolum | 1,348 |
| 14 | TCF Makasli Geyik Yarim_3.Bolum | 1,478 |
| 15 | TCF Spagat_1.Bolum | 570 |
| 16 | TCF Spagat_2.Bolum | 626 |
| 17 | TCF Spagat_3.Bolum | 619 |
| **Total** | | **16,782** |

## Phase Distribution
| Phase | Description | Samples | Percentage |
|-------|-------------|---------|------------|
| 1.Bolum | Starting position | 6,117 | 36.45% |
| 2.Bolum | Mid-movement | 5,216 | 31.08% |
| 3.Bolum | Ending position | 5,449 | 32.47% |

## Data Loss Summary
| Stage | Excluded | Remaining | Reason |
|-------|----------|-----------|--------|
| Initial | 0 | 16,791 | - |
| Zero Detection Filter | 9 | 16,782 | No person detected |
| **Final** | **9** | **16,782** | **0.05% loss** |

## Output Files
| File | Type | Classes | Samples | Purpose |
|------|------|---------|---------|---------|
| gymnastics_complete_dataset.json | JSON | 6 | 16,791 | Raw annotations with all metadata |
| preprocessed_gymnastics_complete_dataset.json | JSON | 6 | 16,782 | Single-person annotations |
| tcf_poses_dataset_vit_6class.csv | CSV | 6 | 16,782 | Training with 6 movement classes |
| tcf_poses_dataset_vit_18class.csv | CSV | 18 | 16,782 | Training with phase-specific classes |

## Summary
The TCF gymnastics dataset was processed from raw images to training-ready CSV files. Person detection was performed with YOLO11x. Pose estimation was performed with ViTPose-H. Path length issues were resolved through folder renaming. Phase information was extracted and corrected. Multiple-person images were reduced to single-person annotations by selecting the highest detection score. Zero-detection images were excluded. The final dataset contains 16,782 annotations across 6 movement classes and 3 phases. Two CSV formats were generated for 6-class and 18-class training scenarios.
