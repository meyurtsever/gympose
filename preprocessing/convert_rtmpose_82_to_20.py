"""
Convert RTMPose 82-class dataset to 20-class dataset.

This script applies the same 82->20 mapping used for VIT dataset
to the RTMPose dataset.
"""
import json
import pandas as pd

# Load the 82-to-20 mapping
with open('output_dataset/82_to_20.json', 'r') as f:
    mapping_82_to_20 = json.load(f)

# Load the 20-class mapping
with open('output_dataset/yoga_poses_dataset_class_mapping_20class.json', 'r') as f:
    class_mapping_20 = json.load(f)

# Create reverse mapping from class name to ID
class_name_to_id = {v: int(k) for k, v in class_mapping_20.items()}

# Load original 82-class RTMPose CSV
df = pd.read_csv('yoga_poses_dataset_rtmpose.csv')

print("="*80)
print("CONVERTING RTMPOSE 82-CLASS TO 20-CLASS")
print("="*80)
print(f"\nOriginal dataset: {len(df)} samples, {df['class_id'].nunique()} classes")

# Load the original 82-class mapping to convert class_id to class_name
with open('output_dataset/yoga_poses_dataset_class_mapping.json', 'r') as f:
    class_mapping_82 = json.load(f)

# Create reverse mapping from 82-class ID to name (mapping is name->id, so reverse it)
id_to_name_82 = {v: k for k, v in class_mapping_82.items()}

# Map each sample from 82-class to 20-class
def map_class_id(old_class_id):
    # Get the old class name
    old_class_name = id_to_name_82[old_class_id]
    # Get the new class name
    new_class_name = mapping_82_to_20[old_class_name]
    # Get the new class ID
    new_class_id = class_name_to_id[new_class_name]
    return new_class_id, new_class_name

# Apply mapping and update both class_id and class_name
df[['class_id', 'class_name']] = df['class_id'].apply(lambda x: pd.Series(map_class_id(x)))

# Save the new 20-class CSV
output_path = 'yoga_poses_dataset_rtmpose_20class.csv'
df.to_csv(output_path, index=False)

print(f"\n20-class dataset saved to: {output_path}")
print(f"Total samples: {len(df)}")
print(f"Number of classes: {df['class_id'].nunique()}")
print("\nClass distribution:")
class_counts = df.groupby(['class_id', 'class_name']).size().reset_index(name='count').sort_values('class_id')
for _, row in class_counts.iterrows():
    print(f"  {row['class_id']:2d} ({row['class_name']}): {row['count']} samples")

print("\n" + "="*80)
print("CONVERSION COMPLETED!")
print("="*80)
