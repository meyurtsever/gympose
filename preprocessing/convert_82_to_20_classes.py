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

# Load original 82-class CSV
df = pd.read_csv('output_dataset/yoga_poses_dataset_vit.csv')

print(f"Original dataset: {len(df)} samples, {df['class_id'].nunique()} classes")

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
    return new_class_id

df['class_id'] = df['class_id'].apply(map_class_id)

# Save the new 20-class CSV
output_path = 'output_dataset/yoga_poses_dataset_vit_20class.csv'
df.to_csv(output_path, index=False)

print(f"\n20-class dataset saved to: {output_path}")
print(f"Total samples: {len(df)}")
print(f"Number of classes: {df['class_id'].nunique()}")
print("\nClass distribution:")
class_counts = df['class_id'].value_counts().sort_index()
for class_id, count in class_counts.items():
    print(f"  {class_id} ({class_mapping_20[str(class_id)]}): {count} samples")
