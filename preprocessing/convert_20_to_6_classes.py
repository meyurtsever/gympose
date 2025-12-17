import pandas as pd
import json
from collections import Counter

# Load the 20-class dataset
print("Loading 20-class dataset...")
df = pd.read_csv('output_dataset/yoga_poses_dataset_vit_20class.csv')
print(f"Loaded {len(df)} samples with {df['class_name'].nunique()} classes")

# Load the 82->20 class mapping first (to convert original class names)
print("\nLoading 82->20 class mapping...")
with open('output_dataset/82_to_20.json', 'r') as f:
    class_82_to_20_mapping = json.load(f)

# Load the 20->6 class mapping
print("Loading 20->6 class mapping...")
with open('output_dataset/20_to_6.json', 'r') as f:
    class_20_to_6_mapping = json.load(f)

# Load the 6-class ID mapping
print("Loading 6-class ID mapping...")
with open('output_dataset/yoga_poses_dataset_class_mapping_6class.json', 'r') as f:
    class_6_id_mapping = json.load(f)

# Create reverse mapping from class name to ID
class_6_name_to_id = {v: int(k) for k, v in class_6_id_mapping.items()}

# Map each row to the new 6-class system
print("\nMapping classes from 82 to 20 to 6...")
# First: 82 original class names -> 20 semantic groups
df['class_name'] = df['class_name'].map(class_82_to_20_mapping)
# Then: 20 semantic groups -> 6 broad categories
df['class_name'] = df['class_name'].map(class_20_to_6_mapping)
df['class_id'] = df['class_name'].map(class_6_name_to_id)

# Verify no missing mappings
if df['class_name'].isna().any() or df['class_id'].isna().any():
    print("ERROR: Some classes could not be mapped!")
    print(df[df['class_name'].isna() | df['class_id'].isna()])
    exit(1)

# Display statistics
print("\n" + "="*60)
print("6-Class Dataset Statistics")
print("="*60)
class_counts = Counter(df['class_name'])
for class_id in sorted(class_6_name_to_id.values()):
    class_name = class_6_id_mapping[str(class_id)]
    count = class_counts[class_name]
    print(f"Class {class_id}: {class_name:20s} - {count:5d} samples")

print(f"\nTotal samples: {len(df)}")
print(f"Total classes: {df['class_name'].nunique()}")

# Save the 6-class dataset
output_path = 'output_dataset/yoga_poses_dataset_vit_6class.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Saved 6-class dataset to: {output_path}")
