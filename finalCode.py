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

# Dataset paths
# These are the paths where our dataset files are stored
DATASET_PATH = "datasets/"
IMAGES_TGZ = os.path.join(DATASET_PATH, "102flowers.tgz")  # Compressed image dataset
EXTRACTED_IMAGES_PATH = os.path.join(DATASET_PATH, "102flowers/jpg")  # Directory to store extracted images
LABELS_MAT = os.path.join(DATASET_PATH, "imagelabels.mat")  # File containing labels for each image
SETID_MAT = os.path.join(DATASET_PATH, "setid.mat")  # File containing training, validation, and test splits

# Extract images if not already extracted
if not os.path.exists(EXTRACTED_IMAGES_PATH):
    print("📦 Extracting images...")  # Will print when the extraction is about to start
    with tarfile.open(IMAGES_TGZ, "r:gz") as tar:  # Open the compressed file
        tar.extractall(path=DATASET_PATH)  # Extract all images to the dataset path
    print(f"✅ Images extracted to: {EXTRACTED_IMAGES_PATH}")  # Confirms extraction

# Load labels and splits
# Labels map each image to a flower category, and splits define training/validation/test sets
# This code is responsible for loading and flattening the labels from a .mat file.
labels = loadmat(LABELS_MAT)["labels"].flatten()  # Load the labels from .mat file and flatten to a 1D array
splits = loadmat(SETID_MAT)  # Load split information
train_ids = splits["trnid"].flatten()  # Training image IDs

# Load images function
def load_images(image_path, ids, labels):
    """
    Load images given their IDs and labels.
    
    Args:
    - image_path: Directory where images are stored.
    - ids: List of image IDs to load.
    - labels: Corresponding labels for the images.

    Returns:
    - images: List of resized images.
    - image_labels: List of labels corresponding to the images.
    """
    images = []
    image_labels = []
    for img_id in ids:
        # Format the image filename with leading zeros (e.g., image_00001.jpg)
        img_name = f"image_{img_id:05d}.jpg"
        img_path = os.path.join(image_path, img_name)
        if os.path.exists(img_path):  # Check if the file exists
            img = cv2.imread(img_path)  # Read the image
            if img is not None:  # Ensure the image was read correctly
                img = cv2.resize(img, (64, 64))  # Resize the image to 64x64 for consistency
                images.append(img)  # Add the image to our list
                image_labels.append(labels[img_id - 1])  # Match 1-based label index to 0-based Python index
    return np.array(images), np.array(image_labels)  # Return as numpy arrays for further processing

# Load training images
print("🚀 Loading training images...")
train_images, train_labels = load_images(EXTRACTED_IMAGES_PATH, train_ids[:500], labels)  # Load first 500 images for visualization
print(f"🎉 Successfully loaded {len(train_images)} training images.")

# Feature extraction: SIFT
def extract_sift_features(images):
    """
    Extract SIFT (Scale-Invariant Feature Transform) features from a list of images.

    Args:
    - images: List of images to process.

    Returns:
    - features: Array of SIFT features for each image.
    """
    sift = cv2.SIFT_create()  # Initialize SIFT feature extractor
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale (SIFT works on grayscale)
        kp, des = sift.detectAndCompute(gray, None)  # Detect keypoints and compute descriptors
        if des is not None:  # If descriptors are found
            features.append(des.flatten()[:128])  # Use only the first 128 features (for consistency)
        else:
            features.append(np.zeros(128))  # If no descriptors, add a zero array
    return np.array(features)

print("🛠️ Extracting SIFT features...")
sift_features = extract_sift_features(train_images)  # Extract shape-related features

# Feature extraction: HSV
def extract_hsv_features(images):
    """
    Extract HSV (Hue, Saturation, Value) color features from a list of images.

    Args:
    - images: List of images to process.

    Returns:
    - features: Array of HSV features (mean values) for each image.
    """
    features = []
    for img in images:
        hsv_img = rgb2hsv(img)  # Convert the image from RGB to HSV color space
        h_mean = np.mean(hsv_img[:, :, 0])  # Compute mean hue
        s_mean = np.mean(hsv_img[:, :, 1])  # Compute mean saturation
        v_mean = np.mean(hsv_img[:, :, 2])  # Compute mean value
        features.append([h_mean, s_mean, v_mean])  # Combine into a feature vector
    return np.array(features)

print("🛠️ Extracting HSV features...")
hsv_features = extract_hsv_features(train_images)  # Extract color-related features

# Dimensionality reduction with Isomap
def apply_isomap(features, n_neighbors=5, n_components=2):
    """
    Reduce the dimensionality of features using Isomap.

    Args:
    - features: High-dimensional feature vectors.
    - n_neighbors: Number of neighbors to consider in Isomap.
    - n_components: Number of dimensions for the output embedding.

    Returns:
    - embedding: Reduced-dimensional embedding.
    - isomap: Trained Isomap model.
    """
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)  # Initialize Isomap
    embedding = isomap.fit_transform(features)  # Fit Isomap and transform features
    return embedding, isomap

# Dimensionality reduction using Isomap
"""
This code reduces the dimensionality of the feature vectors (SIFT and HSV features) to 2D using Isomap, 
which is a manifold learning technique.
"""
print("🚀 Applying Isomap...")
shape_embedding, shape_isomap = apply_isomap(sift_features)  # Isomap for shape features
color_embedding, color_isomap = apply_isomap(hsv_features)  # Isomap for color features

# Visualization with Lines Connecting Images
def plot_embedding_with_images_and_adjusted_lines(embedding, images, n_neighbors, title):
    """
    Visualize an embedding with images connected by lines to their nearest neighbors.

    Args:
    - embedding: 2D coordinates of images after dimensionality reduction.
    - images: List of images to overlay.
    - n_neighbors: Number of neighbors to connect with lines.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 8))  # Set plot size
    ax = plt.gca()  # Get the current axis
    ax.set_title(title)  # Set the plot title
    ax.set_xlabel("Component 1")  # Label x-axis
    ax.set_ylabel("Component 2")  # Label y-axis

    # Compute adjacency matrix using NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)  # Initialize nearest neighbor finder
    neighbors.fit(embedding)  # Fit on the embedding
    adjacency_matrix = neighbors.kneighbors_graph(embedding).toarray()  # Create adjacency matrix

    # Draw lines connecting the nearest neighbors
    for i, (x, y) in enumerate(embedding):
        for j in range(len(embedding)):
            if adjacency_matrix[i, j] > 0:  # Only draw lines for neighbors
                plt.plot([embedding[i, 0], embedding[j, 0]], [embedding[i, 1], embedding[j, 1]], 'b-', alpha=0.5, linewidth=0.5)

    # Overlay images at their corresponding positions
    for i, (x, y) in enumerate(embedding):
        if i < len(images):  # Ensure we don't exceed image count
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)  # Convert image to RGB
            imagebox = OffsetImage(img, zoom=0.4)  # Adjust zoom for smaller images
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)  # Place the image on the plot
            ax.add_artist(ab)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c='black')  # Optional backup dots
    plt.show()  # Display the plot

# Visualize with the refined function
"""
This function is responsible for plotting the 2D embeddings with images and lines connecting nearest neighbors. 
It uses matplotlib to create the visual representation.
"""
print("📊 Visualizing embeddings with improved image connections...")
plot_embedding_with_images_and_adjusted_lines(shape_embedding, train_images[:100], n_neighbors=5, title="Shape Isomap with Image Lines")
plot_embedding_with_images_and_adjusted_lines(color_embedding, train_images[:100], n_neighbors=5, title="Color Isomap with Image Lines")
