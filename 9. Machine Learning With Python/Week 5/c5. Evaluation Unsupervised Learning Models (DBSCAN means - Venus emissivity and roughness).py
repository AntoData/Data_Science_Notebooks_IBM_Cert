import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, \
    davies_bouldin_score
import rasterio
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import Voronoi, voronoi_plot_2d

"""
PROBLEM: We are picking three different GeoTIFF rasters (georeferenced
TIFF files). These are images with embedded spatial metadata
(projection/CRS, pixel size, geographic extent, NoData, etc.)
so any GIS/remote-sensing tool can place them correctly on a map. We
will classify the information about microwave emissivity and meter-scale
surface roughness into clusters and display it over an image of the same
area of Venus's surface

The area these files represent is around 100% of Venus’s surface.
In detail:
Latitude coverage: from −90° to +90° (full global coverage)
Longitude coverage: full 0°–360° domain in simple cylindrical projection
Venus’s total surface area ≈ 460.2 million km² (≈ 4.602 × 10⁸ km²)
Pixel size: 4,641 m per pixel (≈ 22.8 pixels/degree at the equator)

The images are:

Venus_Magellan_MicrowaveEmissivity_Global_4641m.tif
A global map of microwave emissivity for Venus derived from NASA’s
Magellan mission data, published by the USGS Astrogeology Science 
Center.
Emissivity is unitless (0–1) and relates to surface 
dielectric/compositional properties.

SOURCE:
https://planetarymaps.usgs.gov/mosaic/
Venus_Magellan_MicrowaveEmissivity_Global_4641m.tif

Venus_Magellan_MeterScaleSlope_Global_4641m.tif
A global map of meter-scale surface roughness (RMS slope, in degrees)
for Venus derived from Magellan altimetry echo-shape analysis, published
by the USGS Astrogeology Science Center.

SOURCE:
https://planetarymaps.usgs.gov/mosaic/
Venus_Magellan_MeterScaleSlope_Global_4641m.tif

Venus_Magellan_C3-MDIR_Colorized_Global_Mosaic_4641m.tif
A global synthetic-color radar mosaic of Venus created primarily from
Magellan SAR observations, published by the USGS Astrogeology Science 
Center.
Spatial coverage: Entire globe of Venus, spanning latitudes −90° to +90°
and longitudes 0° to 360°, in simple cylindrical (equirectangular) 
projection.

SOURCE:
https://planetarymaps.usgs.gov/mosaic/
Venus_Magellan_C3-MDIR_Colorized_Global_Mosaic_4641m.tif

We will use images Venus_Magellan_MicrowaveEmissivity_Global_4641m.tif
and Venus_Magellan_MeterScaleSlope_Global_4641m.tif to train a DBSCAN
algorithm and classify the different areas of Venus according to their
microwave emissivity and meter-scale surface roughness and then use that
information to display this classification over the image
Venus_Magellan_C3-MDIR_Colorized_Global_Mosaic_4641m.tif. As both refer
to the same area. We will train the DBSCAN model using a sample of the
data and then use those classified points as seeds for a KNeighbours
classifier to complete the classification (as DBSCAN requires more 
resources)
We will try to measure our models using Silhouette scores and
Davies-Bouldin indexes to decide which one is the best for our needs and
get an idea of how accurate our classification is
"""


def check_overlap(bounds1: rasterio.coords.BoundingBox,
                  bounds2: rasterio.coords.BoundingBox) -> bool:
    """
    Checks if the rasterio bounds passed as parameters overlap or not

    :param bounds1: First set of boundaries
    :type bounds1: BoundingBox
    :param bounds2: Second set of boundaries
    :type bounds2: BoundingBox
    :return: True if the boundaries overlap, false otherwise
    :rtype: bool
    """
    return not (
            bounds1.right < bounds2.left or
            bounds1.left > bounds2.right or
            bounds1.top < bounds2.bottom or
            bounds1.bottom > bounds2.top
    )


def colour_scale_array(start_color: (int, int, int),
                       end_color: (int, int, int),
                       steps: int) -> [str]:
    """
    Given two colours and a number it returns an array of  number of
    colours in a scale starting with the first one given and ending with
     the second one given in hexadecimal format as a string

    :param start_color: Array of RGB with the colour where scale starts
    :type start_color: (int, int. int)
    :param end_color: Array of RGB with the colour where scale ends
    :type end_color: (int, int, int)
    :param steps: Number of colours in the scale
    :type steps: int
    :return: Array of colours in hexadecimal format
    :rtype: [str]
    """
    colours: [str] = []
    for step_ in range(steps):
        t: float = step_ / (steps - 1)  # fraction from 0 to 1
        # Interpolate each channel
        r: int = int(start_color[0] + (end_color[0] - start_color[0]) * t)
        g: int = int(start_color[1] + (end_color[1] - start_color[1]) * t)
        b: int = int(start_color[2] + (end_color[2] - start_color[2]) * t)
        # Format as hex code
        colours.append("#{0:02X}{1:02X}{2:02X}".format(r, g, b))
    return colours


def plot_silhouette_scores(silhouette_avg: float,
                           sample_silhouette_values: [float],
                           labels_: [int], performance_summary: str,
                           title_suffix: str = '') -> None:
    """
    Displays the silhouette scores of every point in a model grouped by
    label and compares it to average silhouette score

    :param silhouette_avg: Avg Silhouette score of the model
    :type silhouette_avg: float
    :param sample_silhouette_values: Silhouette scores of every point
    in the variable (these are the points we plot)
    :type sample_silhouette_values: [float]
    :param labels_: Labels assigned to each point in variable x
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
    unique_labels_: [int] = np.unique(labels_)
    colormap: plt.Colormap = plt.get_cmap("tab10")
    color_dict: {int: float} = {
        label: colormap(float(label) / len(unique_labels_))
        for label in unique_labels_}
    y_lower: int = 10
    for ul in unique_labels_:
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


def plot_inertia_sil_score_db_score(k_values_: [int],
                                    clusters_array: [int],
                                    clusters_no_noise_array: [int],
                                    silhouette_scores_: [float],
                                    davies_bouldin_indexes_: [float]) -> None:
    """
    Plots 4 different graphs in the same plot:
    In the first row it displays the number of clusters and the number
    of clusters removing the noise cluster against k
    The bottom row displays the avg silhouette scores and
    Davies Boulding indexes across k
    In all cases k is used in: eps = eps_m0 * k

    :param k_values_: Array with the different k values to be used in
    formula eps = eps_m0 * k
    :type k_values_: [int]
    :param clusters_array: Array with the number of clusters we get
    using the values of k in k_values_
    :type clusters_array: [int]
    :param clusters_no_noise_array: Array with the number of clusters
     we get using the values of k in k_values_, but removing the cluster
     that only contains noise (in case there is such cluster)
    :type clusters_no_noise_array: [int]
    :param silhouette_scores_: Avg Silhouette scores for the values of k
     in k_values
    :type silhouette_scores_: [float]
    :param davies_bouldin_indexes_: Davies Boulding indexes for the
    values of k in k_values
    :type davies_bouldin_indexes_: [float]
    :return: None
    """

    # Plot the inertia values (Elbow Method)
    plt.figure(figsize=(18, 6))

    # Plot silhouette scores
    plt.subplot(2, 2, 1)
    plt.plot(k_values_, clusters_array, marker='o')
    plt.title('Number of clusters vs k')
    plt.xlabel('k*eps_m0 (k)')
    plt.ylabel('Number of clusters')

    # Plot silhouette scores
    plt.subplot(2, 2, 2)
    plt.plot(k_values_, clusters_no_noise_array, marker='o')
    plt.title('Number of clusters without noise vs k')
    plt.xlabel('k*eps_m0 (k)')
    plt.ylabel('Number of clusters without noise')

    # Plot silhouette scores
    plt.subplot(2, 2, 3)
    plt.plot(k_values_, silhouette_scores_, marker='o')
    plt.title('Silhouette Score vs. k')
    plt.xlabel('k*eps_m0 (k)')
    plt.ylabel('Silhouette Score')

    # Plot Davies-Bouldin Index
    plt.subplot(2, 2, 4)
    plt.plot(k_values_, davies_bouldin_indexes_, marker='o')
    plt.title('Davies-Bouldin Index vs. k')
    plt.xlabel('k*eps_m0 (k)')
    plt.ylabel('Davies-Bouldin Index')

    plt.tight_layout()
    plt.show()


def plot_voronoi_2d(var_x: np.ndarray, var_y: np.ndarray,
                    centroids_: np.ndarray) -> None:
    """
    Given a variable x that contains columns x and y with corresponding
     classification labels and an array of centroids
    of those clusters we plot the points and their voronoi 2D diagrams

    :param var_x: Variable x of the model
    :type var_x: np.ndarray
    :param var_y: Labels predicted by the model for variable x
    :type var_y: np.ndarray
    :param centroids_: Centroids of the model
    :type centroids_: np.ndarray

    :return: None
    """

    vor_centroids: np.ndarray = np.unique(centroids_, axis=0)

    print("Creating the voronoi object for the plot")
    voronoi_var: Voronoi = Voronoi(vor_centroids)

    print("Creating the plot")
    fig_, ax = plt.subplots(figsize=(14, 10))

    print("Displaying all our data points")
    colormap = plt.get_cmap("tab10")
    colors_true = colormap(var_y.astype(float) / len(np.unique(var_y)))
    ax.scatter(var_x[:, 0], var_x[:, 1],
               c=colors_true, s=30, alpha=0.3, edgecolors='k',
               label="Data points")

    print("Adding Voronoi 2D diagram")
    voronoi_plot_2d(voronoi_var, ax=ax, show_vertices=False,
                    line_colors='red', line_width=1.5,
                    line_alpha=0.7, point_size=2)

    print("Plotting centroids")
    ax.scatter(vor_centroids[:, 0], vor_centroids[:, 1],
               c='black', marker='*', s=150, label='Centroids')

    ax.set_title('Voronoi of DBSCAN Centroids')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(var_x[:, 0].min() - 0.01, var_x[:, 0].max() +
                0.01)
    ax.set_ylim(var_x[:, 1].min() - 0.01, var_x[:, 1].max() + 0.01)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def sample_for_every_level_idx(
        var_to_sample: np.ndarray, n_points_to_sample: int = 600,
        n_bins: int = 128,
        cap: int | None = None):
    """
    Returns n_points_to_sample indexes sampled from the variable
    var_to_sample in the same proportion as in the original variable
    in n_bins equally separated ranges. Per range, it will return a
    sample which is in the same proportion in those n_points_to_sample
    to return as this range occupies in the whole variable

    :param var_to_sample: Input variable x we want to sample
    :type var_to_sample: np.ndarray
    :param n_points_to_sample: Number of points to sample across
    variable
    :type n_points_to_sample: int
    :param n_bins: Number of equally separated ranges to build to sample
    points in the same proportion as the range occupies in the whole
    variable
    :type n_bins: int
    :param cap: Max number of indexes to return to build the sample
    :type cap: int

    :return: Returns indexes sampled from equal-width bins over the
    first column of var_to_sample, drawing up to per_bin from each
     non-empty bin
    :rtype: np.ndarray
    """
    # Getting max and min values of variable
    x_min: float
    x_max: float
    x_min, x_max = float(np.min(var_to_sample)), float(np.max(var_to_sample))
    # We get n_bins + 1 evenly spaced values in our variable which will
    # become edges of ranges of our classification
    edges: np.ndarray = np.linspace(x_min, x_max, n_bins + 1)

    # We get the indexes of the points with lowest and highest value
    i_min: int = int(np.argmin(var_to_sample))
    i_max: int = int(np.argmax(var_to_sample))
    # Initially we add indexes for points with min and max values
    keep: [] = [i_min, i_max]
    # Numpy random generator
    rng: np.random.Generator = np.random.default_rng(42)
    for b in range(n_bins):
        # We get only the points in our x variable that belong to this
        # range
        sel: np.ndarray = np.where((var_to_sample >= edges[b])
                                   & (var_to_sample < edges[b + 1]))[0]
        # If any point belonged to this range
        if sel.size:
            # We get a sample with a number of points proportional to
            # what this range occupies in the whole variable
            n_samples: int = math.ceil(sel.size * n_points_to_sample /
                                       var_to_sample.size)
            # We pick n_sample points from our variable x that belong
            # to this range
            keep.append(rng.choice(sel, n_samples, replace=False))
    # This take keep which is an array of arrays of indexes and flattens
    # it to 1 dimension, removing possible repeated indexes
    points_indexes: np.ndarray = np.unique(np.concatenate(
        [np.atleast_1d(k_) for k_ in keep]))
    print("Sample size = {0}".format(points_indexes.size))
    # If we set a size to cap our sample and this size is smaller than
    # our current sample
    if cap is not None and points_indexes.size > cap:
        # We pick cap number of points of our indexes
        points_indexes = rng.choice(points_indexes, size=cap, replace=False)
    # We return the indexes of the points we want to use as samples
    return points_indexes


print("1. Opening files")
emm_path: str = "Venus_Magellan_MicrowaveEmissivity_Global_4641m.tif"
rou_path: str = "Venus_Magellan_MeterScaleSlope_Global_4641m.tif"
mos_path: str = "Venus_Magellan_C3-MDIR_Colorized_Global_Mosaic_4641m.tif"

# We open both tif files
with rasterio.open(emm_path) as emm_src, rasterio.open(rou_path) as rou_src, \
        rasterio.open(mos_path) as mos_src:
    # We read the files, we need to read the first band of both tiff
    # files, that method will turn them into a numpy array
    emm_data: np.ndarray = emm_src.read(1).astype(np.float32)
    rou_data: np.ndarray = rou_src.read(1).astype(np.float32)
    mos_band: np.ndarray = mos_src.read(1)

    # Affine transform that tells you how to convert pixel coordinates
    # (row, col) to real-world map coordinates (x, y)
    emm_transform, rou_transform, mos_transform = emm_src.transform, \
        rou_src.transform, mos_src.transform
    # Getting Coordinate Reference System
    emm_crs, rou_crs, mos_crs = emm_src.crs, rou_src.crs, mos_src.crs
    # Getting file bounds
    emm_bounds, rou_bounds, mos_bounds = emm_src.bounds, rou_src.bounds, \
        mos_src.bounds
    mos_shape = (mos_src.height, mos_src.width)

print("emm bounds:", emm_bounds)
print("roughness bounds:", rou_bounds)
print("Mosaic bounds:", mos_bounds)

# We check the boundaries of both files overlap
if not check_overlap(emm_bounds, mos_bounds) or not check_overlap(rou_bounds,
                                                                  mos_bounds):
    raise ValueError("DEM map and NIR mosaic do not overlap!")

print("2. Reproject the datasets so pixels, data match")
print("2.1 Reprojectting dataset emm to dataset mosaic "
      "grid so the area matches")
# Re-projecting dataset DEM to dataset NIR grid so the area matches
emm_resampled = np.empty(mos_shape, dtype=np.float32)
reproject(
    source=emm_data,
    destination=emm_resampled,
    src_transform=emm_transform,
    src_crs=emm_crs,
    dst_transform=mos_transform,
    dst_crs=mos_crs,
    resampling=Resampling.nearest  # discrete categories
)

print("2.2 Re-projecting dataset rou to dataset mosaic "
      "grid so the area matches")
# Re-projecting dataset DEM to dataset NIR grid so the area matches
rou_resampled = np.empty(mos_shape, dtype=np.float32)
reproject(
    source=rou_data,
    destination=rou_resampled,
    src_transform=rou_transform,
    src_crs=rou_crs,
    dst_transform=mos_transform,
    dst_crs=mos_crs,
    resampling=Resampling.nearest  # discrete categories
)

print("2.3 Re-projecting Mosaic image to dataset emm grid so the area matches")
# Re-projecting NIR image to its own grid just to be sure
mos_resampled = np.empty(mos_shape, dtype=mos_band.dtype)
reproject(
    source=mos_band,
    destination=mos_resampled,
    src_transform=mos_transform,
    src_crs=mos_crs,
    dst_transform=mos_transform,
    dst_crs=mos_crs,
    resampling=Resampling.bilinear
)

print("3. Preprocessing data")
print("3.1 Masking invalid data points in emm")
# Masking invalid values in emm
mask_invalid = np.logical_or(emm_resampled <= 0, rou_resampled <= 0)
mask_invalid |= (~np.isfinite(emm_resampled)) | (~np.isfinite(rou_resampled))

emm_resampled = np.ma.masked_array(emm_resampled,
                                   mask=mask_invalid)

print("3.2 Masking invalid data points in rou")
# Masking invalid values in rou
rou_resampled = np.ma.masked_array(rou_resampled,
                                   mask=mask_invalid)
print("3.3 Converting data to physical units")
# convert DNs to physical units
emm_phys = (emm_resampled - 1.0) / 10000.0
rou_phys = (rou_resampled - 1.0) / 10.0
print("3.4 Getting a mask with points whose value is infinite or "
      "magnitude is out of Venus expected bounds and therefore"
      "probably don't reflect real values")
# Masking infinite values
mask_bad = (~np.isfinite(emm_phys)) | (~np.isfinite(rou_phys))
# Adding points with values outside the boundaries for Venus
mask_bad |= (emm_phys < 0.55) | (emm_phys > 1.05) | (rou_phys < 0) | \
            (rou_phys > 60)
print("3.5 Getting mask")
# Getting mask in 2D
valid_2d = ~(mask_invalid | mask_bad)
# Flattening mask
valid_flat = valid_2d.ravel()

print("4. Getting only valid values for emissivity and roughness")
# Use the same mask to index all flattened arrays
emm_ma = np.ma.masked_array(emm_phys, mask=~valid_2d)
rou_ma = np.ma.masked_array(rou_phys, mask=~valid_2d)

print("5. Preparing data")
print("5.1 Joining arrays to get our variable X")
# Flattening physical emissivity and roughness data
emm_flat = emm_phys.ravel()
rou_flat = rou_phys.ravel()
# Adding both arrays to a x variable to train the model
x_train = np.column_stack((emm_flat[valid_flat], rou_flat[valid_flat]))

print("5.2 Applying standard scaler")
std_scr: StandardScaler = StandardScaler()
x_train_scaled: np.ndarray = std_scr.fit_transform(x_train)

# We need to sample our pixels as the Silhouette score is
# computationally expensive
# Sample indices based on one dimension (e.g. emissivity, column 0)
sample_idx: np.ndarray = sample_for_every_level_idx(
    x_train_scaled[:, 0],  # 1D array
    int(x_train_scaled.shape[0] / 800)
)

# Now get the 2D points
sample_var_x: np.ndarray = x_train_scaled[sample_idx, :]  # shape (N, 2)

print("6. Estimating best k in eps = eps_m0 * k")
print("6.1 Applying NearestNeighbors algorithm to start estimating eps0")
# k-distance in the *combined* space
nn: NearestNeighbors = NearestNeighbors(n_neighbors=5,
                                        metric="euclidean").fit(sample_var_x)
print("6.1 We get the distances between points sorted")
dists, _ = nn.kneighbors(sample_var_x, return_distance=True)
k_dist: np.ndarray = dists[:, -1].astype(np.float32)

percentile_k: int = 95
pos: np.ndarray = k_dist > 0
eps0_m: float = np.percentile(k_dist[pos], percentile_k)

print("6.2 We measure our model against a range of values for parameter "
      "k: eps = eps_m0 * k and we will display Silhouette score and "
      "Davies Boulding index but using a sample of our pixels")
k_range: [int] = [k / 100 for k in range(55, 500, 35)]
number_clusters_array: [float] = []
number_clusters_array_no_noise: [float] = []
silhouette_scores: [float] = []
davies_bouldin_scores: [float] = []

# We need to sample our pixels as the Silhouette score is
# computationally expensive


for k in k_range:
    print("For K = {0}".format(k))
    dbscan_measure: DBSCAN = DBSCAN(eps=eps0_m * k, min_samples=50,
                                    metric='euclidean',
                                    algorithm='ball_tree', n_jobs=-1)
    dbscan_measure.fit(sample_var_x)
    print("Number of clusters: ")
    n_clusters: int = len(np.unique(dbscan_measure.labels_))
    number_clusters_array.append(n_clusters)
    if -1 in dbscan_measure.labels_:
        print("That includes a cluster as noise, so it is: {0} ".format(
            n_clusters - 1))
        number_clusters_array_no_noise.append(n_clusters - 1)
    else:
        number_clusters_array_no_noise.append(n_clusters)
    silhouette_avg_: float = silhouette_score(sample_var_x,
                                              dbscan_measure.labels_)
    silhouette_scores.append(silhouette_avg_)
    davies_bouldin_score_: float = davies_bouldin_score(
        sample_var_x, dbscan_measure.labels_)
    davies_bouldin_scores.append(davies_bouldin_score_)

plot_inertia_sil_score_db_score(k_range, number_clusters_array,
                                number_clusters_array_no_noise,
                                silhouette_scores,
                                davies_bouldin_scores)

"""
For DBSCAN, the Silhouette Score and Davies–Bouldin Index provide a 
useful—but limited—way to assess cluster quality across different 
choices of ε (here parameterized as k * eps_m0). Both metrics degrade 
when clusters become ill-defined or when DBSCAN returns mostly noise 
with a single dense core.

Higher silhouette values indicate better separation and cohesion, 
while lower Davies–Bouldin values indicate more compact and distinct 
clusters. In our sweep of k, both metrics change rapidly for small 
values of ε (k ≈ 0.5–1.5), where DBSCAN produces many small, unstable 
clusters. As ε increases, the number of real (non-noise) clusters drops 
quickly, and beyond k ≈ 1.6 the algorithm consistently collapses the 
data into a single cluster. Correspondingly, both the silhouette and 
DBI curves become asymptotic: after this point, increasing ε produces 
only trivial, incremental changes because there is no remaining 
cluster structure for the metrics to evaluate.

This asymptotic behavior is a strong indication that the dataset does 
not possess a clear or robust intrinsic clustering pattern under 
density-based assumptions. The metrics therefore should not be 
interpreted as evidence of strong natural groupings: the “signal” is 
weak, and both measures fluctuate in the small-k region where DBSCAN 
is overly sensitive and returns many fragmented clusters.

Among the ε values that still produce more than one meaningful cluster 
(k ≈ 0.9 and k ≈ 2.3), the silhouette score is markedly higher near 
k = 2.3, suggesting that this region yields the cleanest separation 
between groups before the collapse to a single cluster. Although the 
Davies–Bouldin index is not minimal at this value, DBI is known to be 
unstable for DBSCAN when noise points dominate or when clusters are 
highly unbalanced. Because k = 2.3 produces a small, interpretable 
number of clusters (three), and lies near the optimum of the silhouette 
curve, it represents the best compromise between cluster validity, 
stability, and interpretability for this dataset.
"""
best_k = 2.3
print("8. Creating the DBSCAN algorithm with eps=k*eps_m0 where k={0}".format(
    best_k))
# Apply K-Means
dbscan_mod: DBSCAN = DBSCAN(eps=eps0_m * best_k, min_samples=50,
                            metric='euclidean',
                            algorithm='ball_tree', n_jobs=-1)
print("8.1 Training the model")
dbscan_mod.fit(sample_var_x)
print("9. Getting classification points and centroids")
print("9.1 Getting the classification labels for each pixel")
# Create full labels array, fill valid positions with cluster labels
labels: np.ndarray = dbscan_mod.labels_

print("9.2 Filtering only points in a cluster as variables to train ")
# Use only labeled (non-noise) subset points
mask_labeled = labels != -1
x_train_knn = sample_var_x[mask_labeled]  # shape: (n_samples, 2)
y_train_knn = labels[mask_labeled]

print("10. Applying KNeighborsClassifier to x_train and y_train")
# We apply KNeighborsClassifier to the results we got from applying
# DBSCAN to the samples we worked out previously
knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=5,
                                                 metric="euclidean",
                                                 n_jobs=1)
knn.fit(x_train_knn, y_train_knn)

print("10.1 Predicting labels for ALL pixels using KNeighborsClassifier")
# As working with all pixels at the same time will blow up our RAM
# we will apply a divide and conquer strategy, training the points in
# different groups until all the image is classified
print("Applying a divide and conquer strategy, predicting points by "
      "groups")
# Getting all points and flattening them so we can use them to predict
flat_valid_x: np.ndarray = x_train_scaled.astype(np.float32)
# Number of points we will predict in each interation
step: int = 500_000
# Variable that will contain the cluster of each point in the image
full_labels_valid: np.ndarray = np.empty(
    flat_valid_x.shape[0], dtype=np.int32)
i: int = 1
for s in range(0, flat_valid_x.shape[0], step):
    print("Iteration = {0}".format(i))
    e: int = min(s + step, flat_valid_x.shape[0])
    print("From s={0} to e={1}".format(s, e))
    full_labels_valid[s:e] = knn.predict(flat_valid_x[s:e])
    i += 1

# We create an array full of -1 (represents noise)
labels_full = np.full(emm_flat.shape[0], -1, dtype=np.int32)
# In the points with data, we replace all -1 values by their
# corresponding labels
labels_full[valid_flat] = full_labels_valid.astype(np.int32)
n_clusters: int = np.unique(labels_full[labels_full != -1]).size
print("Number of clusters = {0}".format(n_clusters))

print("10.1 Now we are sorting the labels for the group with the lowest "
      "elevation to the one with the highest")

# --- Sort cluster IDs by emissivity mean, using KNN predictions ---
valid_emm = emm_flat[valid_flat]  # emissivity for valid pixels only
labs = np.unique(full_labels_valid[full_labels_valid != -1])

# mean emissivity per predicted cluster (same valid subset length)
means = [(lab, valid_emm[full_labels_valid == lab].mean())
         for lab in labs]
old_to_new = {old: new for new, (old, _) in enumerate(
    sorted(means, key=lambda t: t[1]))}

# Remap predicted labels to the sorted order
full_labels_valid_sorted = np.vectorize(old_to_new.get)(full_labels_valid)

print("10.2 Sorting the labels")
# Build 2D cluster map for the entire raster (only valid pixels get labels)
cluster_map = np.full(emm_resampled.shape, -1, dtype=np.int16)
cluster_map.ravel()[valid_flat] = full_labels_valid_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("10.3 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
emm_ranges = {}
rou_ranges = {}
for cid in range(len(labs)):
    e_vals = emm_ma[cluster_map == cid]
    r_vals = rou_ma[cluster_map == cid]
    if e_vals.size:
        emm_ranges[cid] = (float(e_vals.min()), float(e_vals.max()))
        rou_ranges[cid] = (float(r_vals.min()), float(r_vals.max()))

print("10.4 Sorting the labels")
# Sort clusters by mean elevation ascending
labels_sorted = np.full_like(labels, -1)
for old, new in old_to_new.items():
    labels_sorted[labels == old] = new

print("10.5 Mapping old labels to the new sorted labels")
# Create a mapping from old label to new sorted label
cluster_map = np.full(emm_phys.shape, -1, np.int16)
labels_sorted = full_labels_valid_sorted  # length == valid_flat.sum()
cluster_map.ravel()[valid_flat] = labels_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("10.6 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
emm_ranges = {}
rou_ranges = {}
for cid in range(n_clusters):
    e_vals = emm_ma[cluster_map == cid]
    r_vals = rou_ma[cluster_map == cid]
    if e_vals.size:
        emm_ranges[cid] = (float(e_vals.min()), float(e_vals.max()))
        rou_ranges[cid] = (float(r_vals.min()), float(r_vals.max()))
print(emm_ranges)
print(rou_ranges)

print("11. Post processing data to display it")
# Rebuilding classification map
# We get valid positions (the ones that are not masked)
valid_positions = np.where(valid_2d)  # ← use the SAME mask you trained with
# We add our labels to valid positions in our new cluster map
# We create a map of the shape of our image full of -1 (invalid points)
cluster_map: np.ndarray = np.full(emm_resampled.shape, -1, dtype=np.int16)
cluster_map[valid_positions] = labels_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)
print("12. Getting emmisivity ranges")
print("\nEmissivity range per cluster:")
for cid in range(n_clusters):
    vals = emm_ma[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .3f} – {vals.max(): .3f}")
print("labels:", labels_sorted.size, "valid pixels:", valid_2d.sum())
print("\nRoughness range per cluster:")
for cid in range(n_clusters):
    vals = rou_ma[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .1f} – {vals.max(): .1f}")

# Define discrete colormap and normalization for labels -1 to best k
colors: [str] = colour_scale_array((252, 201, 192), (77, 21, 6), n_clusters)

cmap: ListedColormap = ListedColormap(colors)
cmap.set_bad(color=(0, 0, 0, 0))

plt.figure(figsize=(12, 10))
plt.imshow(mos_resampled, cmap="gray", aspect="auto")
plt.imshow(cluster_map, cmap=cmap, alpha=0.7, aspect="auto")
plt.title(f"Emissivity and roughness clusters (k={n_clusters}) over Mosaic")

plt.axis('off')
label_names = ["Emi: {0}-{1}, Rou: {2}-{3}ºº".
               format(round(emm_ranges[k_][0], 3), round(emm_ranges[k_][1], 3),
                      round(rou_ranges[k_][0], 3), round(rou_ranges[k_][1], 3))
               for k_ in emm_ranges.keys()]
# Create colorbar with ticks
boundaries = np.arange(-0.5, n_clusters + 0.5, 1)
norm = BoundaryNorm(boundaries, ncolors=n_clusters)
cbar = plt.colorbar(ticks=np.arange(0, n_clusters), boundaries=boundaries)
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)
out_png = r".\DBSCAN_Venus_emissivity_roughness.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig = plt.gcf()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved PNG to: {out_png}")
plt.tight_layout()
plt.show()

print("")
print("Displaying Silhouette scores detailed for a sample")

silhouette_avg_: float = silhouette_score(sample_var_x, dbscan_mod.labels_)
sample_silhouette_values_: np.ndarray = silhouette_samples(sample_var_x,
                                                           dbscan_mod.labels_)
print("Silhouette Avg: = {0}".format(silhouette_avg_))
print("Sample Silhouette values = {0}".format(sample_silhouette_values_))
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
                       dbscan_mod.labels_,
                       silhouette_score_inter,
                       title_suffix=' k-Means Clustering')

# Optional: ignore noise if you kept DBSCAN's -1 label
mask = labels_sorted != -1
x_points_used = flat_valid_x[mask]
labels_used = labels_sorted[mask]

unique_clusters = np.unique(labels_used)

centroids = np.vstack([
    x_points_used[labels_used == c].mean(axis=0)
    for c in unique_clusters
])
plot_voronoi_2d(sample_var_x,
                dbscan_mod.labels_, centroids)
