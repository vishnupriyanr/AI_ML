import os
import tarfile
import numpy as np
import cv2
from scipy.io import loadmat

# Dataset paths
DATASET_PATH = "datasets/"
IMAGES_TGZ = os.path.join(DATASET_PATH, "102flowers.tgz")
EXTRACTED_IMAGES_PATH = os.path.join(DATASET_PATH, "102flowers/jpg")
LABELS_MAT = os.path.join(DATASET_PATH, "imagelabels.mat")
SETID_MAT = os.path.join(DATASET_PATH, "setid.mat")

# Extract images if not already extracted
if not os.path.exists(EXTRACTED_IMAGES_PATH):
    print("📦 Extracting images...")
    with tarfile.open(IMAGES_TGZ, "r:gz") as tar:
        tar.extractall(path=DATASET_PATH)
    print(f"✅ Images extracted to: {EXTRACTED_IMAGES_PATH}")

# Verify the number of extracted files and list some of them
extracted_files = os.listdir(EXTRACTED_IMAGES_PATH)
print(f"📂 Number of extracted files: {len(extracted_files)}")
print(f"🖼️ Sample extracted files: {extracted_files[:10]}")  # List first 10 files

# Load labels and dataset splits
print("📊 Loading labels and splits...")
labels = loadmat(LABELS_MAT)["labels"].flatten()

splits = loadmat(SETID_MAT)
train_ids = splits["trnid"].flatten()
val_ids = splits["valid"].flatten()
test_ids = splits["tstid"].flatten()

# Debugging: Verify dataset structure
print(f"📝 Labels shape: {labels.shape}, Sample labels: {labels[:10]}")
print(f"📊 Training set size: {len(train_ids)}, Validation set size: {len(val_ids)}, Test set size: {len(test_ids)}")
print(f"🔍 First 10 Training IDs: {train_ids[:10]}")

# Debug: Check if training image paths exist
print("🛠️ Verifying first 10 training image paths...")
for img_id in train_ids[:10]:  
    img_name = f"image_{img_id:05d}.jpg"  
    img_path = os.path.join(EXTRACTED_IMAGES_PATH, img_name)
    print(f"🔍 Checking: {img_path}, Exists: {os.path.exists(img_path)}")

# ✅ Define load_images function BEFORE calling it
def load_images(image_path, ids, labels):
    images = []
    image_labels = []
    for img_id in ids:
        img_name = f"image_{img_id:05d}.jpg"  
        img_path = os.path.join(image_path, img_name)
        if os.path.exists(img_path):
            print(f"✅ Loading: {img_path}")
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for consistency
                images.append(img)
                image_labels.append(labels[img_id - 1])  # Adjust 1-based index to 0-based labels
            else:
                print(f"⚠️ Image could not be loaded: {img_path}")
        else:
            print(f"❌ File not found: {img_path}")
    return np.array(images), np.array(image_labels)

# Now load training images
print("🚀 Loading training images...")
train_images, train_labels = load_images(EXTRACTED_IMAGES_PATH, train_ids, labels)
print(f"🎉 Successfully loaded {len(train_images)} training images.")
