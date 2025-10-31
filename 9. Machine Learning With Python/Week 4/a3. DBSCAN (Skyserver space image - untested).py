import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

"""
SOURCE: Image comes from: 
https://skyserver.sdss.org/dr19/VisualTools/navi

The goal is to classify the pixels on the image to identify the
celestial objects in the object and isolate them from the dark
background and classify those elements depending on their intensity
"""

print("1. Opening the image")
# We use imread (image read) from skimage
image = imread('skyserver.png')

print("2. Preprocessing the image")
print("2.1 We reshape the image if it has 4 channels RGB and brightness")
if image.shape[-1] == 4:
    image = image[:, :, :3]

print("2.2 Turning the image into a pixels array")
# Reshape for clustering
pixels = image.reshape(-1, 3)
pixels_x = pixels.astype(np.float32)
print("3. Applying the elbow method to get best k")
inertias = []
k_values = range(1, 10)  # Try k from 1 to 9

# k-distance in the *combined* space
nn = NearestNeighbors(n_neighbors=5,
                      metric="euclidean").fit(pixels)
print("6.1 We get the distances between points sorted")
kdist = np.sort(
    nn.kneighbors(pixels))

print("6.2 Getting percentile 90 of distances between points")
eps0_m = np.percentile(kdist, 80)
k = 1
print("4. Creating the Kmeans algorithm with k={0}".format(k))
# Apply K-Means
print("7.1 Multiplier k = {0}".format(k))
eps_m = float(eps0_m * k)
print("7.2 Building the DBSCAN object")
dbscan_mod: DBSCAN = DBSCAN(min_samples=5, eps=eps_m, metric="euclidean",
                            n_jobs=1)
print("7.3 Training the model")
dbscan_mod.fit(pixels_x)
print("7.4 Getting labels for each point")
labels = dbscan_mod.labels_

centers = (pixels[pixels.label != -1]
           .groupby("label")[["latitude", "longitude"]]
           .mean())

print("7. Masking the pixel representing the dark space")
# Compute brightness of each cluster center
print("7.1 Getting the brightness of every centroid")
brightness = np.linalg.norm(centers, axis=1)  # or use np.mean(centers, axis=1)

print("7.2 Getting the darkest centroid")
# Index of the darkest cluster
dark_cluster_idx = np.argmin(brightness)

print("7.3 Rebuilding the segmented image using labels")
# Reconstruct segmented image
segmented_image = labels.reshape(image.shape[:2])

print("7.4 Masking the points in the darkest cluster")
masked = np.copy(segmented_image)
masked[segmented_image == dark_cluster_idx] = -1  # Mask dark area

print("8. Plotting the original image")
# Plot
plt.figure(figsize=(16, 10))  # Wider and taller window
plt.title("Original Image")
plt.imshow(image)
plt.tight_layout()
plt.show()

print("9. Plotting the segmented image")
# Optional: set a colormap that skips the masked cluster
plt.figure(figsize=(16, 10))
plt.imshow(masked, cmap='plasma')  # Mask will appear as black
plt.title('Clusters Without Dark Background')
plt.axis('off')
plt.show()
