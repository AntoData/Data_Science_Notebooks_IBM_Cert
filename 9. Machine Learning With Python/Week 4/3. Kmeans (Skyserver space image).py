import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from kneed import KneeLocator

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

print("3. Applying the elbow method to get best k")
inertias = []
k_values = range(1, 10)  # Try k from 1 to 9

for k in k_values:
    print("3.1 Training the Kmeans algorithm with k={0}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++")
    kmeans.fit(pixels)
    print("3.2 Adding inertia for this k to the array")
    inertias.append(kmeans.inertia_)

print("3.3 Plotting the inertias")
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Choosing k")
plt.grid(True)
plt.show()

print("3.4 Applying KneeLocator to get our best k according to the method")
kl: KneeLocator = \
    KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
best_k: int = kl.elbow
print("Best k (elbow found at):", best_k)

print("4. Creating the Kmeans algorithm with k={0}".format(best_k))
# Apply K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42, init="k-means++")
print("5. Training the model")
kmeans.fit(pixels)
print("6. Getting classification points and centroids")
labels = kmeans.labels_
centers = kmeans.cluster_centers_

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
