import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from sklearn.metrics import silhouette_score, silhouette_samples, \
    davies_bouldin_score

"""
SOURCE: Image comes from: 
https://skyserver.sdss.org/dr19/VisualTools/navi

The goal is to classify the pixels on the image to identify the
celestial objects in the object and isolate them from the dark
background and classify those elements depending on their intensity
"""


def plot_silhouette_scores(silhouette_avg: float,
                           sample_silhouette_values: [float],
                           labels_: [int], performance_summary: str,
                           title_suffix: str = '') -> None:
    """
    Displays the silhouette scores of every point in a model by grouped
    label and compares it to average silhouette score

    :param silhouette_avg: Avg Silhouette score of the model
    :type silhouette_avg: float
    :param sample_silhouette_values: Silhouette scores of every point
    in the variable (these are the points we plot)
    :type sample_silhouette_values: [float]
    :param labels_: Labels assigned to each point in variable X
    :type labels_: [int]
    :param performance_summary: Summary of how the model performs to be
    displayed as title
    :type performance_summary: str
    :param title_suffix: Suffix to add to the title of the plot
    :type title_suffix: str
    :return: None
    """

    ax = plt.gca()  # Get the current axis if none is provided

    # Plot silhouette analysis on the provided axis
    unique_labels: [int] = np.unique(labels_)
    colormap: plt.Colormap = plt.get_cmap("tab10")
    color_dict: {int: float} = {
        label: colormap(float(label) / len(unique_labels))
        for label in unique_labels}
    y_lower: int = 10
    for ul in unique_labels:
        ith_cluster_silhouette_values: [float] = \
            sample_silhouette_values[labels_ == ul]
        ith_cluster_silhouette_values.sort()
        size_cluster_i: int = ith_cluster_silhouette_values.shape[0]
        y_upper: int = y_lower + size_cluster_i
        color = color_dict[ul]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(ul))
        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Score for {title_suffix} \n' +
                 f'Average Silhouette: {silhouette_avg: .2f}\n' +
                 performance_summary)
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.25, 1])  # Set the x-axis range to [0, 1]

    ax.set_yticks([])
    plt.show()


def plot_inertia_sil_score_db_score(k_values_: [int], inertias_: [float],
                                    silhouette_scores_: [float],
                                    davies_bouldin_indexes_: [float]) -> None:
    """
    Plots in 3 different graphs the inertia, avg silhouette scores and
    Davies Bouldin indexes across K (number of clusters in Kmeans)

    :param k_values_: Array with the different number of clusters that
    were used in the experiment
    :type k_values_: [int]
    :param inertias_: Array with the inertia values we got for the K
    values in the previous variable
    :type inertias_: [float]
    :param silhouette_scores_: Avg Silhouette scores for the K values in
    k_values in a Kmeans model
    :type silhouette_scores_: [float]
    :param davies_bouldin_indexes_: Davies Bouldin indexes for the K
    values in k_values in a Kmeans model
    :type davies_bouldin_indexes_: [float]
    :return: None
    """

    # Plot the inertia values (Elbow Method)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(k_values_, inertias_, marker='o')
    plt.title('Elbow Method: Inertia vs. k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    # Plot silhouette scores
    plt.subplot(1, 3, 2)
    plt.plot(k_values_, silhouette_scores_, marker='o')
    plt.title('Silhouette Score vs. k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    # Plot Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(k_values_, davies_bouldin_indexes_, marker='o')
    plt.title('Davies-Bouldin Index vs. k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Index')

    plt.tight_layout()
    plt.show()


print("1. Opening the image")
# We use imread (image read) from skimage
image: np.ndarray = imread('skyserver.png')

print("2. Preprocessing the image")
print("2.1 We reshape the image if it has 4 channels RGB and brightness")
if image.shape[-1] == 4:
    image = image[:, :, :3]

print("2.2 Turning the image into a pixels array")
# Reshape for clustering
pixels: np.ndarray = image.reshape(-1, 3)

print("3. Applying the elbow method to get best k")
inertias: [] = []
k_values: [int] = range(1, 10)  # Try k from 1 to 9

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

print("4.1 We repeat with a higher range and add Silhouette score and "
      "Davies Bouldin index but with a sample of our pixels")
k_range: [int] = [k for k in range(2, 21)]
inertia: [float] = []
silhouette_scores: [float] = []
davies_bouldin_scores: [float] = []

# We need to sample our pixels as the Silhouette score is
# computationally expensive
# First, we sample rows of the pixels
idx: [int] = np.random.choice(pixels.shape[0], size=int(pixels.shape[0] / 25),
                              replace=False)
# Finally, we sample the pixels using those row indexes
sample_pixels: np.ndarray = pixels[idx]

for k in k_range:
    print("For K = {0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    kmeans.fit(sample_pixels)
    inertia.append(kmeans.inertia_)
    silhouette_avg_: float = silhouette_score(sample_pixels,
                                              kmeans.labels_)
    silhouette_scores.append(silhouette_avg_)
    davies_bouldin_score_: float = davies_bouldin_score(sample_pixels,
                                                        kmeans.labels_)
    davies_bouldin_scores.append(davies_bouldin_score_)

plot_inertia_sil_score_db_score(k_range, inertia, silhouette_scores,
                                davies_bouldin_scores)

"""
Inertia measures the total within-cluster sum of squared distances 
— lower is better.
However, inertia always decreases as you add clusters, so you look for 
the “elbow point” — where adding more clusters doesn't give a big 
improvement.
In this case, we can see the biggest improvement happens between 
k ≈ 2–3, after that the improvement is not that big

Higher silhouette = better cluster separation and cohesion.
It typically peaks near the “best” number of clusters.
In this case, we see the best value is clearly k = 2 with around 0.9, 
after that the silhouette scores are much lower

Lower DBI = better clustering (clusters are compact and well-separated).
DBI increases sharply after K = 2 

Clearly, the only valid k value is K=2 as it will divide the image 
between two classes, background and stars
"""

print("5. Creating the Kmeans algorithm with k={0}".format(2))
# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42, init="k-means++")
print("5.1 Training the model")
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
out_png = r".\skyserver_k2.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig = plt.gcf()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved PNG to: {out_png}")
plt.show()

print("")
print("Displaying Silhouette scores detailed for a sample")
kmeans_sc: KMeans = KMeans(init="k-means++", n_clusters=2, n_init=12)
kmeans_sc.fit(sample_pixels)
silhouette_avg_: float = silhouette_score(sample_pixels, kmeans_sc.labels_)
sample_silhouette_values_: [float] = silhouette_samples(
    sample_pixels, kmeans_sc.labels_)
print("Silhouette Avg: = {0}".format(silhouette_avg_))
print("Sample Silhouette values = {0}".format(sample_silhouette_values_))

silhouette_score_inter: str = ""
if silhouette_avg_ > 0.7:
    silhouette_score_inter = "Very strong structure, " \
                             "clusters are well separated and tight."
elif 0.5 < silhouette_avg_ <= 0.7:
    silhouette_score_inter = "Good structure, reasonably " \
                             "distinct clusters."
elif 0.25 < silhouette_avg_ <= 0.5:
    silhouette_score_inter = "Moderate structure, " \
                             "clusters overlap somewhat."
elif silhouette_avg_ <= 0.25:
    silhouette_score_inter = "Poor structure, clustering may not " \
                             "be meaningful."

print(silhouette_score_inter)
plot_silhouette_scores(silhouette_avg_, sample_silhouette_values_,
                       kmeans_sc.labels_,
                       silhouette_score_inter,
                       title_suffix=' k-Means Clustering')
