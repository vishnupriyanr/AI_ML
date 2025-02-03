import os
import tarfile
import numpy as np
import cv2
from scipy.io import loadmat
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

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

# Load labels and splits
labels = loadmat(LABELS_MAT)["labels"].flatten()
splits = loadmat(SETID_MAT)
train_ids = splits["trnid"].flatten()

# Load images function
def load_images(image_path, ids, labels):
    images = []
    image_labels = []
    for img_id in ids:
        img_name = f"image_{img_id:05d}.jpg"  # Ensure correct naming
        img_path = os.path.join(image_path, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for consistency
                images.append(img)
                image_labels.append(labels[img_id - 1])  # Match 1-based index to 0-based labels
    return np.array(images), np.array(image_labels)

# Load training images
print("🚀 Loading training images...")
train_images, train_labels = load_images(EXTRACTED_IMAGES_PATH, train_ids[:1020], labels)  # Use a subset for efficiency
print(f"🎉 Successfully loaded {len(train_images)} training images.")

# Feature extraction: SIFT
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None:
            features.append(des.flatten()[:128])  # Use first 128 features
        else:
            features.append(np.zeros(128))  # Handle empty descriptors
    return np.array(features)

print("🛠️ Extracting SIFT features...")
sift_features = extract_sift_features(train_images)

# Feature extraction: HSV
def extract_hsv_features(images):
    features = []
    for img in images:
        hsv_img = rgb2hsv(img)
        h_mean = np.mean(hsv_img[:, :, 0])  # Hue
        s_mean = np.mean(hsv_img[:, :, 1])  # Saturation
        v_mean = np.mean(hsv_img[:, :, 2])  # Value
        features.append([h_mean, s_mean, v_mean])
    return np.array(features)

print("🛠️ Extracting HSV features...")
hsv_features = extract_hsv_features(train_images)

# Dimensionality reduction with Isomap
def apply_isomap(features, n_neighbors=5, n_components=2):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    embedding = isomap.fit_transform(features)
    return embedding

print("🚀 Applying Isomap...")
shape_embedding = apply_isomap(sift_features)
color_embedding = apply_isomap(hsv_features)

# Visualization
def plot_embedding(embedding, images, title):
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(embedding):
        plt.scatter(x, y, s=30, edgecolors="k")
        if i < len(images):
            img = images[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img, extent=(x - 0.3, x + 0.3, y - 0.3, y + 0.3), zorder=10)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

print("📊 Visualizing embeddings...")
plot_embedding(shape_embedding, train_images[:100], "Shape Isomap")
plot_embedding(color_embedding, train_images[:100], "Color Isomap")
