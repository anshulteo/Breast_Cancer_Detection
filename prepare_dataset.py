import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import random

# Configuration
ORIGINAL_DATASET_PATH = 'original_dataset'
OUTPUT_DATASET_PATH = 'dataset'
TEST_SIZE = 0.2  # 20% for test, 80% for train
RANDOM_STATE = 42

# Classes
CLASSES = ['benign', 'malignant', 'normal']

def create_directories():
    """Create train and test directory structure"""
    print("Creating directory structure...")
    
    for split in ['train', 'test']:
        for class_name in CLASSES:
            path = os.path.join(OUTPUT_DATASET_PATH, split, class_name)
            os.makedirs(path, exist_ok=True)
    
    print("✅ Directories created successfully!")

def split_and_copy_images():
    """Split images into train and test sets"""
    print("\n" + "="*50)
    print("Starting dataset split...")
    print("="*50)
    
    for class_name in CLASSES:
        source_path = os.path.join(ORIGINAL_DATASET_PATH, class_name)
        
        # Check if source directory exists
        if not os.path.exists(source_path):
            print(f"⚠️  Warning: Directory '{source_path}' not found. Skipping...")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(Path(source_path).glob(f'*{ext}'))
        
        all_images = [str(img) for img in all_images]
        
        if len(all_images) == 0:
            print(f"⚠️  Warning: No images found in '{source_path}'. Skipping...")
            continue
        
        print(f"\n📁 Processing class: {class_name.upper()}")
        print(f"   Total images: {len(all_images)}")
        
        # Split into train and test
        train_images, test_images = train_test_split(
            all_images,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )
        
        print(f"   Training images: {len(train_images)}")
        print(f"   Testing images: {len(test_images)}")
        
        # Copy train images
        train_dest = os.path.join(OUTPUT_DATASET_PATH, 'train', class_name)
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(train_dest, img_name)
            shutil.copy2(img_path, dest_path)
        
        # Copy test images
        test_dest = os.path.join(OUTPUT_DATASET_PATH, 'test', class_name)
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(test_dest, img_name)
            shutil.copy2(img_path, dest_path)
        
        print(f"   ✅ Copied successfully!")
    
    print("\n" + "="*50)
    print("Dataset split completed!")
    print("="*50)

def print_summary():
    """Print summary of the dataset"""
    print("\n📊 DATASET SUMMARY")
    print("="*50)
    
    total_train = 0
    total_test = 0
    
    for split in ['train', 'test']:
        print(f"\n{split.upper()} SET:")
        for class_name in CLASSES:
            path = os.path.join(OUTPUT_DATASET_PATH, split, class_name)
            if os.path.exists(path):
                count = len(os.listdir(path))
                print(f"  {class_name.capitalize()}: {count} images")
                
                if split == 'train':
                    total_train += count
                else:
                    total_test += count
    
    print("\n" + "="*50)
    print(f"Total Training Images: {total_train}")
    print(f"Total Testing Images: {total_test}")
    print(f"Total Images: {total_train + total_test}")
    print("="*50)

def verify_original_dataset():
    """Verify that original dataset exists and has images"""
    print("Checking original dataset...")
    
    if not os.path.exists(ORIGINAL_DATASET_PATH):
        print(f"\n❌ ERROR: '{ORIGINAL_DATASET_PATH}' directory not found!")
        print(f"\nPlease create the following structure:")
        print(f"\n{ORIGINAL_DATASET_PATH}/")
        print(f"  ├── benign/")
        print(f"  ├── malignant/")
        print(f"  └── normal/")
        print(f"\nAnd place your images in the respective folders.")
        return False
    
    found_images = False
    for class_name in CLASSES:
        class_path = os.path.join(ORIGINAL_DATASET_PATH, class_name)
        if os.path.exists(class_path) and len(os.listdir(class_path)) > 0:
            found_images = True
            break
    
    if not found_images:
        print(f"\n❌ ERROR: No images found in '{ORIGINAL_DATASET_PATH}'!")
        return False
    
    print("✅ Original dataset verified!")
    return True

def main():
    print("="*50)
    print("BREAST CANCER DATASET PREPARATION")
    print("="*50)
    
    # Verify original dataset exists
    if not verify_original_dataset():
        return
    
    # Clean previous dataset if exists
    if os.path.exists(OUTPUT_DATASET_PATH):
        print(f"\n⚠️  '{OUTPUT_DATASET_PATH}' already exists.")
        response = input("Do you want to delete and recreate it? (yes/no): ").lower()
        if response == 'yes':
            shutil.rmtree(OUTPUT_DATASET_PATH)
            print("✅ Previous dataset removed.")
        else:
            print("❌ Operation cancelled.")
            return
    
    # Create directories
    create_directories()
    
    # Split and copy images
    split_and_copy_images()
    
    # Print summary
    print_summary()
    
    print("\n✅ Dataset preparation complete!")
    print(f"You can now run: python train_model.py")

if __name__ == "__main__":
    main()
