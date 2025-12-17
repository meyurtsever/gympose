"""
Convert yoga poses JSON annotations to CSV format for model training.

This script:
1. Reads yoga_poses_complete_dataset.json
2. Extracts keypoints for each annotation
3. Skips records with empty or multiple keypoint arrays
4. Creates CSV with columns: class_name, nose_x, nose_y, left_eye_x, left_eye_y, ...
5. Encodes class names for classification
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

def analyze_json_structure(json_path):
    """Analyze the JSON file structure."""
    print("=" * 80)
    print("ANALYZING JSON STRUCTURE")
    print("=" * 80)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nTop-level keys: {list(data.keys())}")
    
    if 'skeleton' in data:
        print(f"\nSkeleton keypoints: {data['skeleton']}")
        print(f"Total keypoints: {len(data['skeleton'])}")
    
    if 'annotations' in data:
        print(f"\nTotal annotations: {len(data['annotations'])}")
        
        # Sample first annotation
        if len(data['annotations']) > 0:
            print(f"\nSample annotation structure:")
            sample = data['annotations'][0]
            print(f"  Keys: {list(sample.keys())}")
            
            if 'keypoints' in sample:
                print(f"  Keypoints type: {type(sample['keypoints'])}")
                if isinstance(sample['keypoints'], list):
                    print(f"  Keypoints length: {len(sample['keypoints'])}")
                    if len(sample['keypoints']) > 0:
                        print(f"  First keypoint array shape: {np.array(sample['keypoints'][0]).shape if len(sample['keypoints']) > 0 else 'N/A'}")
    
    # Count statistics
    if 'annotations' in data:
        empty_keypoints = 0
        multiple_keypoints = 0
        single_keypoints = 0
        classes = set()
        
        for ann in data['annotations']:
            if 'class_name' in ann:
                classes.add(ann['class_name'])
            
            if 'keypoints' in ann:
                if len(ann['keypoints']) == 0:
                    empty_keypoints += 1
                elif len(ann['keypoints']) == 1:
                    single_keypoints += 1
                else:
                    multiple_keypoints += 1
        
        print(f"\nStatistics:")
        print(f"  Unique classes: {len(classes)}")
        print(f"  Empty keypoints: {empty_keypoints}")
        print(f"  Single person (valid): {single_keypoints}")
        print(f"  Multiple people (skip): {multiple_keypoints}")
        print(f"  Valid annotations: {single_keypoints}")
    
    return data

def convert_to_csv(json_path, output_csv_path):
    """Convert JSON annotations to CSV format."""
    print("\n" + "=" * 80)
    print("CONVERTING TO CSV")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load JSON data
    print(f"\nLoading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    
    print(f"Total annotations: {len(annotations)}")
    
    # Get skeleton from first valid annotation
    skeleton_dict = None
    for ann in annotations:
        if 'skeleton' in ann and ann['skeleton']:
            skeleton_dict = ann['skeleton']
            break
    
    if skeleton_dict is None:
        print("ERROR: No skeleton found in annotations!")
        return None, None
    
    # Convert skeleton dict to ordered list
    skeleton = [skeleton_dict[str(i)] for i in range(len(skeleton_dict))]
    print(f"Skeleton has {len(skeleton)} keypoints: {skeleton}")
    
    # Prepare data for CSV
    rows = []
    skipped_empty = 0
    skipped_multiple = 0
    
    for idx, ann in enumerate(annotations):
        class_name = ann.get('class_name', 'unknown')
        keypoints = ann.get('keypoints', [])
        image_path = ann.get('image_path', '')
        
        # Skip if keypoints is empty
        if len(keypoints) == 0:
            skipped_empty += 1
            continue
        
        # Keypoints format: [{"0": [[x, y, conf], ...], "1": [[x, y, conf], ...]}, ...]
        # Each dict represents detected people, keys are person IDs
        
        # Get first item which is a dict of person_id -> keypoints
        keypoints_dict = keypoints[0] if isinstance(keypoints[0], dict) else {}
        
        # Skip if empty dict (no detections)
        if not keypoints_dict:
            skipped_empty += 1
            continue
        
        # Skip if multiple people detected
        if len(keypoints_dict) > 1:
            skipped_multiple += 1
            continue
        
        # Get the single person's keypoints (first and only key)
        person_id = list(keypoints_dict.keys())[0]
        person_keypoints = keypoints_dict[person_id]  # List of [x, y, conf] for each keypoint
        
        # Create row dictionary
        row = {
            'class_name': class_name,
            'image_path': image_path
        }
        
        # Add keypoint coordinates
        for kp_idx, kp_name in enumerate(skeleton):
            if kp_idx < len(person_keypoints):
                x, y, conf = person_keypoints[kp_idx]
                # Clean keypoint name for column
                clean_name = kp_name.replace(' ', '_').replace('-', '_').lower()
                row[f'{clean_name}_x'] = x
                row[f'{clean_name}_y'] = y
                row[f'{clean_name}_conf'] = conf
            else:
                # If keypoint is missing, set to NaN
                clean_name = kp_name.replace(' ', '_').replace('-', '_').lower()
                row[f'{clean_name}_x'] = np.nan
                row[f'{clean_name}_y'] = np.nan
                row[f'{clean_name}_conf'] = np.nan
        
        rows.append(row)
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(annotations)} annotations...")
    
    # Create DataFrame
    print(f"\nCreating DataFrame with {len(rows)} valid rows...")
    df = pd.DataFrame(rows)
    
    # Encode class names
    print(f"\nEncoding class names...")
    class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(df['class_name'].unique()))}
    df['class_id'] = df['class_name'].map(class_mapping)
    
    # Reorder columns: class_id, class_name, image_path, then keypoints
    cols = ['class_id', 'class_name', 'image_path'] + [col for col in df.columns if col not in ['class_id', 'class_name', 'image_path']]
    df = df[cols]
    
    # Save to CSV
    print(f"\nSaving to CSV: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    
    # Save class mapping
    mapping_path = output_csv_path.replace('.csv', '_class_mapping.json')
    print(f"Saving class mapping to: {mapping_path}")
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total annotations processed: {len(annotations)}")
    print(f"Skipped (empty keypoints): {skipped_empty}")
    print(f"Skipped (multiple people): {skipped_multiple}")
    print(f"Valid rows in CSV: {len(rows)}")
    print(f"Number of classes: {len(class_mapping)}")
    print(f"Number of keypoints: {len(skeleton)}")
    print(f"Total columns in CSV: {len(df.columns)}")
    print(f"\nClass mapping (first 10):")
    for class_name, class_id in list(class_mapping.items())[:10]:
        print(f"  {class_id}: {class_name}")
    if len(class_mapping) > 10:
        print(f"  ... and {len(class_mapping) - 10} more")
    
    print(f"\nCSV shape: {df.shape}")
    print(f"Conversion time: {elapsed_time:.2f} seconds")
    print(f"\nFiles created:")
    print(f"  - {output_csv_path}")
    print(f"  - {mapping_path}")
    
    return df, class_mapping

if __name__ == "__main__":
    # Paths
    json_path = "output_dataset/yoga_poses_complete_dataset.json"
    output_csv = "output_dataset/yoga_poses_dataset.csv"
    
    # Analyze structure first
    data = analyze_json_structure(json_path)
    
    # Convert to CSV
    df, class_mapping = convert_to_csv(json_path, output_csv)
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
