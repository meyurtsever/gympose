"""
Prepare 18-class CSV for training by copying combined_class_id to class_id column.

This script creates a training-ready version of the 18-class RTMPose CSV where:
- class_id is replaced with combined_class_id values (0-17)
- All other columns remain unchanged
"""

import pandas as pd
from pathlib import Path

def prepare_18class_csv():
    """Prepare 18-class CSV for training."""
    
    input_path = "TCF_DATASET/tcf_pose_keypoints_rtmpose_18class.csv"
    output_path = "TCF_DATASET/tcf_pose_keypoints_rtmpose_18class_training.csv"
    
    print(f"Loading CSV from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"\nOriginal CSV:")
    print(f"  Total rows: {len(df)}")
    print(f"  class_id unique values: {sorted(df['class_id'].unique())}")
    print(f"  combined_class_id unique values: {sorted(df['combined_class_id'].unique())}")
    
    # Replace class_id with combined_class_id
    df['class_id'] = df['combined_class_id']
    
    print(f"\nModified CSV:")
    print(f"  class_id unique values: {sorted(df['class_id'].unique())}")
    print(f"  Number of classes: {len(df['class_id'].unique())}")
    
    # Save the modified CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved training-ready CSV to: {output_path}")
    
    # Verify the saved file
    df_verify = pd.read_csv(output_path)
    print(f"\nVerification:")
    print(f"  Total rows: {len(df_verify)}")
    print(f"  class_id range: {df_verify['class_id'].min()} to {df_verify['class_id'].max()}")
    print(f"  Number of classes: {df_verify['class_id'].nunique()}")
    
    return output_path

if __name__ == "__main__":
    output_path = prepare_18class_csv()
    print(f"\n{'='*60}")
    print(f"✓✓✓ CSV READY FOR TRAINING ✓✓✓")
    print(f"{'='*60}")
    print(f"\nUse this file for 18-class training:")
    print(f"  {output_path}")
    print(f"\nExample commands:")
    print(f"  python average_train_xgboost_classifier.py --data {output_path}")
    print(f"  python average_train_resnext50_classifier.py --data {output_path}")
    print(f"  python average_train_densenet121_classifier.py --data {output_path}")
    print(f"  python average_train_fttransformer_classifier.py --data {output_path}")
