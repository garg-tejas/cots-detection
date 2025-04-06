import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# Set up paths for our dataset
BASE_DIR = "tensorflow-great-barrier-reef/"
OUTPUT_DIR = "reef_yolo"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Create the necessary directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

# Load annotation data
df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
df = df[df['annotations'] != '[]']

# Split into training (80%) and validation (20%) sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def process_split(split_df, split_name):
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
        img_name = row['image_id']
        video_id, frame_id = img_name.split('-')
        img_path = os.path.join(BASE_DIR, 'train_images', f"video_{video_id}", f"{frame_id}.jpg")

        # Copy the image to our dataset directory
        shutil.copy(img_path, os.path.join(IMAGES_DIR, split_name, f"{img_name}.jpg"))

        # Parse and convert annotations to YOLO format
        annots = json.loads(row['annotations'].replace("'", '"'))

        label_path = os.path.join(LABELS_DIR, split_name, f"{img_name}.txt")
        with open(label_path, "w") as f:
            for annot in annots:
                # Convert to normalized YOLO coordinates
                x_center = (annot['x'] + annot['width'] / 2) / 1280
                y_center = (annot['y'] + annot['height'] / 2) / 720
                width = annot['width'] / 1280
                height = annot['height'] / 720
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

process_split(train_df, 'train')
process_split(val_df, 'val')

print("✅ Dataset prepared for YOLOv8!")