# Keypoint Structure

## COCO-WholeBody Format

This project uses the COCO-WholeBody keypoint format for pose annotations. The following tables describe the keypoint indices and corresponding body parts.

### ViTPose: 25 Keypoints

| Index | Body Part | Description |
|-------|-----------|-------------|
| 0 | Nose | Center of the face |
| 1 | Left Eye | Left eye center |
| 2 | Right Eye | Right eye center |
| 3 | Left Ear | Left ear position |
| 4 | Right Ear | Right ear position |
| 5 | Neck | Neck/shoulder junction |
| 6 | Left Shoulder | Left shoulder joint |
| 7 | Right Shoulder | Right shoulder joint |
| 8 | Left Elbow | Left elbow joint |
| 9 | Right Elbow | Right elbow joint |
| 10 | Left Wrist | Left wrist joint |
| 11 | Right Wrist | Right wrist joint |
| 12 | Left Hip | Left hip joint |
| 13 | Right Hip | Right hip joint |
| 14 | Hip Center | Center point between hips |
| 15 | Left Knee | Left knee joint |
| 16 | Right Knee | Right knee joint |
| 17 | Left Ankle | Left ankle joint |
| 18 | Right Ankle | Right ankle joint |
| 19 | Left Big Toe | Left big toe tip |
| 20 | Left Small Toe | Left small toe tip |
| 21 | Left Heel | Left heel position |
| 22 | Right Big Toe | Right big toe tip |
| 23 | Right Small Toe | Right small toe tip |
| 24 | Right Heel | Right heel position |

### RTMPose: 26 Keypoints

RTMPose extends the ViTPose format with one additional keypoint (index 25), providing enhanced anatomical coverage.

## Keypoint Representation

Each keypoint is represented by three values:
- **x**: Horizontal pixel coordinate
- **y**: Vertical pixel coordinate  
- **confidence**: Detection confidence score in range [0, 1]

## CSV Data Format

In the CSV files, keypoints are stored as:
```
kp0_x, kp0_y, kp0_conf, kp1_x, kp1_y, kp1_conf, ..., kp24_x, kp24_y, kp24_conf
```

For training, only the (x, y) coordinates are used, resulting in:
- **50-dimensional feature vectors** for ViTPose (25 keypoints × 2 coordinates)
- **52-dimensional feature vectors** for RTMPose (26 keypoints × 2 coordinates)

## Skeleton Connections

The keypoints form a skeletal structure with the following connections:
- **Head**: Nose → Eyes → Ears
- **Torso**: Neck → Shoulders → Hip Center → Hips
- **Arms**: Shoulders → Elbows → Wrists
- **Legs**: Hips → Knees → Ankles → Toes/Heels

## Usage in Classification

The classification models use only the normalized (x, y) coordinate pairs as input features. Confidence scores are discarded during training to prevent the model from learning pose estimation quality rather than movement patterns.
