import os
import tarfile
import numpy as np
import cv2
from scipy.io import loadmat
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.color import rgb2hsv
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
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
                img = cv2.resize(img, (64, 64))  # Resize for smaller thumbnails
                images.append(img)
                image_labels.append(labels[img_id - 1])  # Match 1-based index to 0-based labels
    return np.array(images), np.array(image_labels)

# Load training images
print("🚀 Loading training images...")
train_images, train_labels = load_images(EXTRACTED_IMAGES_PATH, train_ids[:500], labels)  # Limit to 500 for visualization
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
    return embedding, isomap

print("🚀 Applying Isomap...")
shape_embedding, shape_isomap = apply_isomap(sift_features)
color_embedding, color_isomap = apply_isomap(hsv_features)

# Visualization with Lines Connecting Images
# Improved visualization function to ensure lines connect images directly
def plot_embedding_with_images_and_adjusted_lines(embedding, images, n_neighbors, title):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Compute adjacency matrix using NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(embedding)
    adjacency_matrix = neighbors.kneighbors_graph(embedding).toarray()

    # Draw lines connecting the nearest neighbors
    for i, (x, y) in enumerate(embedding):
        for j in range(len(embedding)):
            if adjacency_matrix[i, j] > 0:  # Only draw lines for neighbors
                plt.plot([embedding[i, 0], embedding[j, 0]],
                         [embedding[i, 1], embedding[j, 1]],
                         'b-', alpha=0.5, linewidth=0.5)  # Lighter, thinner lines for clarity

    # Overlay images at their corresponding positions
    for i, (x, y) in enumerate(embedding):
        if i < len(images):
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            imagebox = OffsetImage(img, zoom=0.4)  # Adjust zoom to reduce overlap
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c='black')  # Optional backup dots
    plt.show()

# Visualize with the refined function
print("📊 Visualizing embeddings with improved image connections...")
plot_embedding_with_images_and_adjusted_lines(shape_embedding, train_images[:100], n_neighbors=5, title="Shape Isomap with Image Lines")
plot_embedding_with_images_and_adjusted_lines(color_embedding, train_images[:100], n_neighbors=5, title="Color Isomap with Image Lines")