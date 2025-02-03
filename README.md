# AI_ML

This repository contains the code for the AI and ML project.

Run finalCode.py
The other files are process python files
The final results are in images t4_fig_1.png and t4_fig_2.png
Get datasets from here: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

What happens in the code, quick jist:

Load the dataset.
Compute SIFT features for shape descriptors.
Use HSV for color descriptors.
Visualize the dataset using isomap embeddings.



RESULT IMAGES

Images show feature clustering
If two images are close in the plot, it means their features (shape or color) are similar.
If two images are far apart, their features differ significantly.

Feature importance
The Shape Isomap groups flowers with similar contours and edge structures (e.g., flowers with circular petals might cluster together).
The Color Isomap clusters flowers based on their dominant colors (e.g., all yellow flowers might appear close together).

Class overlap
Looking at the scatter patterns to evaluate whether there is overlap between flower categories.
Overlapping points suggest that those categories share similar features.


These visualizations can help one understand how well the features separate the categories.
They can guide the selection of features for machine learning models (e.g., shape features for flowers with unique structures, color features for visually distinct flowers).

WHAT IT TELLS

Shape Isomap
The wide spread and slight overlap in clusters suggest that shape features vary widely across the dataset.
Flowers with similar shapes (e.g., round versus pointed petals) tend to cluster together.

Color Isomap
The clusters are more distinct because color is often a strong differentiator for flowers.
Flowers with predominantly similar hues (e.g., orange, purple) are grouped.

IMAGES

The key difference between the t2 visualizations (with points and images) and the reference visualizations (with lines connecting images) lies in how the relationships between data points are represented.

Points represent individual images in the dataset, plotted in a 2D space based on their Isomap embeddings.
No lines mean that the relationships between images (or the distances between them) are not explicitly visualized.
The focus is purely on the positions of images in the embedding space, showing how similar or dissimilar they are in terms of shape (SIFT) or color (HSV).

Lines represent the neighborhood graph that Isomap uses to compute the low-dimensional embeddings.
Isomap connects each data point (image) to its k nearest neighbors in the high-dimensional space, forming a graph. This graph is then used to compute pairwise distances for dimensionality reduction.
The lines visualize the connectivity between points (images), showing which images are considered similar in the Isomap graph.

T2 images versus the reference images:

With Lines (Neighborhood Graph):
Shows explicit relationships between images.
Helps you understand which images are considered similar in the high-dimensional feature space.
Useful for analyzing the structure of the data and the Isomap embeddings.
Without Lines (Only Points):
Focuses purely on the 2D arrangement of images.
Highlights clusters and groups of similar images without emphasizing specific pairwise connections.

Lines help to:
Show how the embedding was constructed (i.e., which images were neighbors in the original feature space).
Highlight potential clusters and relationships between nearby points.

T3 images:

Lines:

What do they connect?
In the T3 plots, the lines connect the points (black dots) in the embedded 2D Isomap space. These points represent the underlying position of the feature embeddings for each image (either SIFT features for shape or HSV features for color).
The images are overlaid on these points as visual annotations, but the lines do not directly connect the images. They are based on the proximity of the feature embeddings, determined by the Isomap algorithm.
What do they mean?
The lines represent the nearest-neighbor relationships between feature embeddings in the reduced Isomap space. Essentially, the Isomap algorithm constructs a graph where each point is connected to its closest neighbors in the high-dimensional feature space. The lines in the T3 plots visualize this graph.


Lines in the Reference Plots:
What do they connect?
In the reference plots, the lines visually connect the images themselves. This means the graphical connections are drawn based on the relative positions of the images in the Isomap embedding space.
What do they mean?
The lines in the reference indicate similarities in feature space (shape or color descriptors). By directly connecting images, the visualization emphasizes the visual or categorical relationships between specific flower images.

T3 Plots (Connecting Points):
The focus is on the abstract embedding space and the graph structure of the feature embeddings. The images are secondary, providing a visual cue for each point.
This method is more analytical, showing the geometric relationships and connections as determined by Isomap.
Reference Plots (Connecting Images):
The focus is on the relationships between the actual flower images, providing a more visually intuitive representation.
This emphasizes the perceptual or categorical grouping of flowers, making it easier to see clusters of similar images.



