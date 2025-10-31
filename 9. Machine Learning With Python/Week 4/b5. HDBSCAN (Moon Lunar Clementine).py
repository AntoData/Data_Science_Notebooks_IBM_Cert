import math
import os
import random as rd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from hdbscan import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

"""
PROBLEM: We are picking two different GeoTIFF rasters (georeferenced 
TIFF files). These are images with embedded spatial metadata 
(projection/CRS, pixel size, geographic extent, NoData, etc.) 
so any GIS/remote-sensing tool can place them correctly on a map. We 
will classify the information about iron oxide (FeO) into clusters 
and display it over an image of the same area of the Moon

The area these files represent is around 88% of the Moon's surface. 
In detail:
70°S to 70°N at 1 km/pixel resolution.
Latitude coverage: 140° (out of 180° total)
Longitude coverage: likely full 360°
Approximate global area of Moon: 37.9 million km²
Area between 70°S and 70°N ≈ ~33.4 million km² (about 88% of lunar 
surface)

The images are:

-   Lunar_Clementine_NIR_cal_empcor_500m.tif
    A global mosaic from Clementine's near‑infrared (NIR) camera, 
    covering six spectral bands: 1100, 1250, 1500, 2000, 2600, and 
    2780 nm 
    Processed to 500 m/pixel resolution in a simple cylindrical 
    (equirectangular) projection, spanning latitudes −70° to +70° and
     longitudes −180° to +180° (or 0°–360° in some releases)
    SOURCE:
    https://planetarymaps.usgs.gov/mosaic/
    Lunar_Clementine_NIR_cal_empcor_500m.tif

-   Lunar_Clementine_UVVIS_FeO_ClrBinned_70S70N_1km.tif
    Derived from the Clementine UV/Visible camera data.
    Represents iron oxide (FeO) abundance expressed as weight percent.
    The map is a binned color product, where discrete FeO ranges 
    (e.g. 0–25%) are grouped into color categories and rendered as 
    discrete color bands
    SOURCE:
    https://planetarymaps.usgs.gov/mosaic/
    Lunar_Clementine_UVVIS_FeO_ClrBinned_70S70N_1km.tif

We will use image Lunar_Clementine_UVVIS_FeO_ClrBinned_70S70N_1km to get
a sample we can use to train a HDBSCAN algorithm, as this algorithm is 
 very costly and classify the different areas of the Moon
 according to their FeO concentration. We will use these clustered 
 points as seeds for a KNeighborsClassifier where we will classify the 
 rest of the points in the image by parts in different iterations 
 applying a Divide and Conquer strategy as this will blow up our RAM 
 otherwise. Finally, we will use that information  to display this 
 classification over the image Lunar_Clementine_NIR_cal_empcor_500m as 
 both refer to the same area
"""


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


print("1. Opening files")
feo_path: str = "Lunar_Clementine_UVVIS_FeO_ClrBinned_70S70N_1km.tif"
nir_path: str = "Lunar_Clementine_NIR_cal_empcor_500m.tif"

# We open both tif files
with rasterio.open(feo_path) as feo_src, rasterio.open(nir_path) as nir_src:
    # We read the files, we need to read the first band of both tiff
    # files, that method will turn them into a numpy array
    feo_data: np.ndarray = feo_src.read(1).astype(np.float32)
    nir_band: np.ndarray = nir_src.read(1)

    # Affine transform that tells you how to convert pixel coordinates
    # (row, col) to real-world map coordinates (x, y)
    feo_transform, nir_transform = feo_src.transform, nir_src.transform
    # Getting Coordinate Reference System
    feo_crs, nir_crs = feo_src.crs, nir_src.crs
    # Getting file bounds
    feo_bounds, nir_bounds = feo_src.bounds, nir_src.bounds
    nir_shape = (nir_src.height, nir_src.width)

print("FeO bounds:", feo_bounds)
print("NIR bounds:", nir_bounds)

# We check the boundaries of both files overlap
if not check_overlap(feo_bounds, nir_bounds):
    raise ValueError("FeO map and NIR mosaic do not overlap!")

print("2. Reproject the datasets so pixels, data match")
print("2.1 Re-projecting dataset FeO to dataset NIR grid so the area matches")
# Re-projecting dataset FeO to dataset NIR grid so the area matches
feo_resampled: np.ndarray = np.empty(nir_shape, dtype=np.float32)
reproject(
    source=feo_data,
    destination=feo_resampled,
    src_transform=feo_transform,
    src_crs=feo_crs,
    dst_transform=nir_transform,
    dst_crs=nir_crs,
    resampling=Resampling.nearest  # discrete categories
)

print("2.2 Re-projecting NIR image to dataset FeO grid so the area matches")
# Reproject NIR image to its own grid just to be sure
nir_resampled: np.ndarray = np.empty(nir_shape, dtype=nir_band.dtype)
reproject(
    source=nir_band,
    destination=nir_resampled,
    src_transform=nir_transform,
    src_crs=nir_crs,
    dst_transform=nir_transform,
    dst_crs=nir_crs,
    resampling=Resampling.bilinear
)

print("3. Preprocessing data")
print("3.1 Masking invalid data points in FeO")
# Masking invalid values in FeO
mask_invalid: np.ndarray = feo_resampled < 0
feo_resampled: np.ma.masked_array = np.ma.masked_array(feo_resampled,
                                                       mask=mask_invalid)

print("3.2 Flattening our data to use it later to train")
# Use the same mask to index all flattened arrays
flat_feo: np.ndarray = feo_resampled.data.ravel()
flat_mask: np.ndarray = ~feo_resampled.mask.ravel()

# Apply mask
valid_feo_values: np.ndarray = flat_feo[flat_mask].reshape(-1, 1)

print("3.3 Checking we have at least 10 samples")
if valid_feo_values.shape[0] < 10:
    raise ValueError("Too few valid FeO samples to cluster.")

print("4. Sampling our variable x")
# HDBSCAN is very costly for this amount of points so we will select a
# number of random points to train our HDBSCAN algorithm. These points
# will be the seed for a KNearestNeighbor algorithm which will apply
# the last part of the classification

# However we will use the method defined and the beginning of the file
# to make sure our sample's points are representative of our variable

# We get the number of elements of our variable x
n_points: int = valid_feo_values.shape[0]
# We set the size of our samples
eps_subset_size: int = min(2_000_000, n_points)  # for eps estimation
clust_subset_size: int = min(2_000_000, n_points)  # for HDBSCAN itself

# We get the indexes to apply to our variable x to build the samples to
# get an estimation of eps and apply HDBSCAN to
eps_idx: np.ndarray = sample_for_every_level_idx(
    valid_feo_values, n_points_to_sample=100_000, n_bins=128)
clust_idx: np.ndarray = sample_for_every_level_idx(
    valid_feo_values, n_points_to_sample=100_000, n_bins=128)

# Now we build the samples of our variable x
x_sample_eps: np.ndarray = valid_feo_values[eps_idx]
x_sample_labels: np.ndarray = valid_feo_values[clust_idx]

print("5. Applying NearestNeighbors algorithm to start estimating eps0")
# k-distance in the *combined* space
nn: NearestNeighbors = NearestNeighbors(n_neighbors=5,
                                        metric="euclidean").fit(x_sample_eps)
print("5.1 We get the distances between points sorted")
dists, _ = nn.kneighbors(x_sample_eps, return_distance=True)
k_dist: np.ndarray = dists[:, -1].astype(np.float32)

percentile_k: int = 95
print("5.2 Getting percentile {0} of distances between points".format(
    percentile_k))
print("We will use it to get esp0_m")
k: float = 1.0
print("k = {0}".format(k))
print("eps_m will be eps0_m * k")

# Ignore zero k-distances (caused by identical FeO values)
eps0_m: float = 0.0
eps_m: float = 0.0
pos: np.ndarray = k_dist > 0
if np.any(pos):
    eps0_m = np.percentile(k_dist[pos], percentile_k)
    eps_m = float(eps0_m * k)
else:
    # All k-distances are zero -> values are discretized.
    # Fall back to half the smallest non-zero gap between unique values.
    vals = np.unique(x_sample_eps.ravel())
    if vals.size >= 2:
        gaps = np.diff(np.sort(vals))
        nz = gaps[gaps > 0]
        if nz.size:
            eps_m = 0.5 * float(nz.min())
        else:
            raise ValueError("FeO values are all identical; HDBSCAN "
                             "cannot form clusters.")
    else:
        raise ValueError("FeO values are all identical; HDBSCAN cannot "
                         "form clusters.")
print("eps_m = {0}".format(eps_m))
print("6. Creating the HDBSCAN algorithm with k={0}".format(k))
print("6.1 Building the HDBSCAN object")
hdbscan_mod: HDBSCAN = HDBSCAN(cluster_selection_epsilon=eps_m,
                               min_samples=10, min_cluster_size=10,
                               metric='euclidean',
                               cluster_selection_method="eom",
                               prediction_data=True)
print("6.2 Training the model")
hdbscan_mod.fit(x_sample_labels)

print("7. Getting the classification labels for each pixel")
# Create full labels array, fill valid positions with cluster labels
labels: np.ndarray = hdbscan_mod.labels_

print("7.1 Filtering only points in a cluster as variables to train ")
# Use only labeled (non-noise) subset points
mask_labeled: np.ndarray = labels != -1
x_train: np.ndarray = x_sample_labels[mask_labeled].reshape(-1)
y_train: np.ndarray = labels[mask_labeled]

print("8. Applying KNeighborsClassifier to x_train and y_train")
# We apply KNeighborsClassifier to the results we got from applying
# HDBSCAN to the samples we worked out previously
knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=5,
                                                 metric="euclidean",
                                                 n_jobs=1)
knn.fit(x_train.reshape(-1, 1), y_train)

print("8.1 Predicting labels for ALL pixels using KNeighborsClassifier")
# As working with all pixels at the same time will blow up our RAM
# we will apply a divide and conquer strategy, training the points in
# different groups until all the image is classified
print("Applying a divide and conquer strategy, predicting points by "
      "groups")
# Getting all points and flattening them so we can use them to predict
flat_valid_feo: np.ndarray = valid_feo_values.reshape(-1, 1).astype(np.float32)
# Number of points we will predict in each interation
step: int = 500_000
# Variable that will contain the cluster of each point in the image
full_labels_valid: np.ndarray = np.empty(
    flat_valid_feo.shape[0], dtype=np.int32)
i: int = 1
for s in range(0, flat_valid_feo.shape[0], step):
    print("Iteration = {0}".format(i))
    e: int = min(s + step, flat_valid_feo.shape[0])
    print("From s={0} to e={1}".format(s, e))
    full_labels_valid[s:e] = knn.predict(flat_valid_feo[s:e])
    i += 1

# We create an array full of -1 (represents noise)
labels_full = np.full(flat_feo.shape[0], -1)
# In the points with data, we replace all -1 values by their
# corresponding labels
labels_full[flat_mask] = full_labels_valid.astype(np.int32)
n_clusters: int = np.unique(labels_full[labels_full != -1]).size
print("Number of clusters = {0}".format(n_clusters))

print("9.1 Now we are sorting the labels for the group with the lowest "
      "FeO to the one with the highest")
# We get an array with the list of labels (unique labels)
unique_labels: np.ndarray = np.unique(labels_full)
# To store the mean of the elevation in a cluster
mean_feos: [float] = []
# Key will be the label and value a tuple with min and max elevation
feo_ranges: dict = {}
# For each label
for lab in unique_labels:
    # We get the mean of the elevation and store it
    mean_feo = float(valid_feo_values[labels_full == lab].mean())
    mean_feos.append((lab, mean_feo))

print("9.2 Sorting the labels by mean FeO")
# Sort clusters by mean elevation ascending
sorted_clusters: [] = sorted(mean_feos, key=lambda x: x[1])
print("9.3 Mapping old labels to the new labels sorted by FeO")
# Create a mapping from old label to new sorted label
old_to_new_label: dict = {old_lab: new_lab for new_lab, (old_lab, _) in
                          enumerate(sorted_clusters)}

print("9.4 Now we replace the old labels for the new sorted labels in "
      "the classified image")
# Remap labels array (keeping nodata as -1)
labels_sorted_full = np.full(labels_full.shape, -1, dtype=int)
for old_lab, new_lab in old_to_new_label.items():
    labels_sorted_full[labels_full == old_lab] = new_lab

print("9.5 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
unique_labels_sorted = np.unique(labels_sorted_full)
for lab in unique_labels:
    # We turn dem_data into an array and select only the pixels with
    # the corresponding label
    feo_values: np.ndarray = valid_feo_values.flatten()[
        labels_sorted_full == lab]
    # Now we get the min and max elevations in the cluster and store
    # it in the dictionary
    feo_ranges[lab] = (np.min(feo_values), np.max(feo_values))
print(feo_ranges)

print("10. Post processing data to display it")
# Rebuilding classification map
# We create a map of the shape of our image full of -1 (invalid points)
cluster_map: np.ndarray = np.full(feo_resampled.shape, -1, dtype=np.int16)
cluster_map[~mask_invalid] = labels_sorted_full
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("11. Getting FeO ranges")
print("\nFeO range per cluster:")
for cid in range(n_clusters):
    vals = feo_resampled[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .2f} – {vals.max(): .2f}")

# Define discrete colormap and normalization for labels -1 to best k
colors: [str] = ['#{:02x}{:02x}{:02x}'.format(rd.randint(64, 255),
                                              rd.randint(64, 255),
                                              rd.randint(64, 255))
                 for _ in range(0, n_clusters)]
cmap: ListedColormap = ListedColormap(colors)
cmap.set_bad(color=(0, 0, 0, 0))
boundaries = np.arange(-0.5, n_clusters + 0.5, 1)
norm = BoundaryNorm(boundaries, ncolors=n_clusters)
plt.figure(figsize=(12, 10))
plt.imshow(nir_resampled, cmap="gray", aspect="auto")
plt.imshow(cluster_map, cmap=cmap, alpha=0.3, aspect="auto", norm=norm)
plt.title(f"Clementine FeO Clusters (k={n_clusters}) over NIR Mosaic")

plt.axis('off')
label_names = ["From {0} to {1}".format(elev[0],
                                        elev[1])
               for elev in
               feo_ranges.values()]
# Create colorbar with ticks
cbar = plt.colorbar(ticks=np.arange(0, n_clusters), boundaries=boundaries)
# Set custom tick labels
out_png = r".\HDBSCAN_Moon_Lunar_Clementine_reworked_sample.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig = plt.gcf()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved PNG to: {out_png}")
cbar.ax.set_yticklabels(label_names)
plt.tight_layout()
plt.show()
