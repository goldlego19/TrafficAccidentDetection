import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION ---
# Where your images are currently sitting
SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'optical_flow_mapsNEW')
# Where you want the new structured dataset to be built
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'split_mapsNEW')

# How much data goes into each folder
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # Must add up to 1.0!

def split_dataset():
    # Find your classes (e.g., 'accident', 'normal')
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found classes: {classes}")
    print("Building new folder structure...")

    # Create the train, val, and test folders for every class
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

    # Process each class one by one
    for cls in classes:
        cls_dir = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(cls_dir)
        
        # Shuffle the images randomly
        random.shuffle(images)
        
        # Calculate exactly how many images go into each split
        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        # Slice the list of images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # We use shutil.copy so your original folder is kept safe as a backup
        print(f"\nCopying {cls} images...")
        
        for img in tqdm(train_images, desc=f"{cls} -> Train", leave=False):
            shutil.copy(os.path.join(cls_dir, img), os.path.join(DEST_DIR, 'train', cls, img))
            
        for img in tqdm(val_images, desc=f"{cls} -> Val  ", leave=False):
            shutil.copy(os.path.join(cls_dir, img), os.path.join(DEST_DIR, 'val', cls, img))
            
        for img in tqdm(test_images, desc=f"{cls} -> Test ", leave=False):
            shutil.copy(os.path.join(cls_dir, img), os.path.join(DEST_DIR, 'test', cls, img))

    print(f"\n✅ Dataset successfully split and saved to: {DEST_DIR}")

if __name__ == '__main__':
    split_dataset()