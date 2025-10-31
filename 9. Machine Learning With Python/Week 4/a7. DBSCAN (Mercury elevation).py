import os
import math
import numpy as np
import rasterio
from kneed import KneeLocator
from rasterio.warp import reproject, Resampling
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

"""
PROBLEM: We are picking two different GeoTIFF rasters (georeferenced 
TIFF files). These are images with embedded spatial metadata 
(projection/CRS, pixel size, geographic extent, NoData, etc.) 
so any GIS/remote-sensing tool can place them correctly on a map. We 
will classify the information about elevation of the terrain (DEM) 
into clusters and display it over an image of the same area of Mercury's 
surface

The area these files represent is around 100% of Mercury’s surface.
In detail:
Latitude coverage: from −90° to +90° (full global coverage)
Longitude coverage: full 0°–360° domain in simple cylindrical projection
Mercury’s total surface area ≈ 74.8 million km² (≈ 7.48 × 10⁷ km²)
Pixel size: 665 m per pixel (≈ 167 pixels/degree at the equator)

The images are:

Mercury_MESSENGER_GLD100_Elevation_Global_665m_v2.tif
A global digital elevation model (DEM) of Mercury produced from stereo
 and photoclinometry data collected by NASA’s MESSENGER spacecraft, 
 published by the USGS Astrogeology Science Center.
Elevation is derived relative to Mercury’s reference sphere 
(mean radius ≈ 2,439.7 km).

SOURCE:
https://planetarymaps.usgs.gov/mosaic/
Mercury_MESSENGER_GLD100_Elevation_Global_665m_v2.tif

Mercury_MESSENGER_Global_Mosaic_665m.tif
A global monochrome image mosaic of Mercury, created from observations
 by the Mercury Dual Imaging System (MDIS) aboard NASA’s MESSENGER 
 spacecraft.
Published by the USGS Astrogeology Science Center as part of the global
 mapping of Mercury from MESSENGER data products.
Spatial coverage: Entire globe of Mercury, spanning latitudes
 −90° to +90° and longitudes 0° to 360°, in simple cylindrical 
 (equirectangular) projection.
 
SOURCE:
https://planetarymaps.usgs.gov/mosaic/
Mercury_MESSENGER_Global_Mosaic_665m.tif

We will use image Mercury_MESSENGER_GLD100_Elevation_Global_665m_v2.tif 
to create a proportional sample of the variables. Then, we will use that
 sample to train a DBSCAN algorithm and use its labels as the seeds of 
 a KNeighborsClassifier to classify the different areas of Mercury 
according to their terrain elevation and then use that information to 
display this classification over the image 
Mercury_MESSENGER_Global_Mosaic_665m.tif. As both refer to the same area
We will use a Divide and Conquer strategy as predicting the labels for 
the whole image at once will blow up our RAM so we will get the 
predictions in batches
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


def colour_scale_array(start_color: (int, int, int),
                       end_color: (int, int, int),
                       steps: int) -> [str]:
    """
    Given two colours and a number it returns an array that number of
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
    for i in range(steps):
        t: float = i / (steps - 1)  # fraction from 0 to 1
        # Interpolate each channel
        r: int = int(start_color[0] + (end_color[0] - start_color[0]) * t)
        g: int = int(start_color[1] + (end_color[1] - start_color[1]) * t)
        b: int = int(start_color[2] + (end_color[2] - start_color[2]) * t)
        # Format as hex code
        colours.append(f"#{r:02X}{g:02X}{b:02X}")
    return colours


print("1. Opening files")
dem_path: str = "Mercury_Messenger_USGS_DEM_Global_665m_v2.tif"
mos_path: str = "Mercury_Messenger_MDIS_Basemap_EnhancedColor_Mosaic_" \
                "Global_665m.tif"

# We open both tif files
with rasterio.open(dem_path) as dem_src, rasterio.open(mos_path) as mos_src:
    # We read the files, we need to read the first band of both tiff
    # files, that method will turn them into a numpy array
    dem_data: np.ndarray = dem_src.read(1).astype(np.float32)
    mos_band: np.ndarray = mos_src.read(1)

    # Affine transform that tells you how to convert pixel coordinates
    # (row, col) to real-world map coordinates (x, y)
    dem_transform, mos_transform = dem_src.transform, mos_src.transform
    # Getting Coordinate Reference System
    dem_crs, mos_crs = dem_src.crs, mos_src.crs
    # Getting file bounds
    dem_bounds, mos_bounds = dem_src.bounds, mos_src.bounds
    mos_shape = (mos_src.height, mos_src.width)

print("DEM bounds:", dem_bounds)
print("Mosaic bounds:", mos_bounds)

# We check the boundaries of both files overlap
if not check_overlap(dem_bounds, mos_bounds):
    raise ValueError("DEM map and NIR mosaic do not overlap!")

print("2. Reproject the datasets so pixels, data match")
print("2.1 Re-projecting dataset DEM to dataset mosaic "
      "grid so the area matches")
# Re-projecting dataset DEM to dataset NIR grid so the area matches
dem_resampled = np.empty(mos_shape, dtype=np.float32)
reproject(
    source=dem_data,
    destination=dem_resampled,
    src_transform=dem_transform,
    src_crs=dem_crs,
    dst_transform=mos_transform,
    dst_crs=mos_crs,
    resampling=Resampling.nearest  # discrete categories
)

print("3. Re-projecting Mosaic image to dataset DEM grid so the area matches")
# Reproject NIR image to its own grid just to be sure
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

print("4. Preprocessing data")
print("4.1 Masking invalid data points in DEM")
# Masking invalid values in DEM
mask_invalid = dem_resampled < 0
dem_resampled = np.ma.masked_array(dem_resampled,
                                   mask=mask_invalid)

print("5. Flattening our data to use it later to train")
# Use the same mask to index all flattened arrays
flat_dem = dem_resampled.data.ravel()
flat_mask: np.ndarray = ~dem_resampled.mask.ravel()

# Apply mask
valid_dem_values: np.ndarray = flat_dem[flat_mask].reshape(-1, 1)

print("6. Checking we have at least 10 samples")
if valid_dem_values.shape[0] < 10:
    raise ValueError("Too few valid DEM samples to cluster.")


print("7. Sampling our variable x")
# DBSCAN is very costly for this amount of points so we will select a
# number of random points to train our DBSCAN algorithm. These points
# will be the seed for a KNearestNeighbor algorithm which will apply
# the last part of the classification

# However we will use the method defined and the beginning of the file
# to make sure our sample's points are representative of our variable

# We get the number of elements of our variable x
n_points: int = valid_dem_values.shape[0]
# We set the size of our samples
eps_subset_size: int = min(2_000_000, n_points)  # for eps estimation
clust_subset_size: int = min(2_000_000, n_points)  # for DBSCAN itself

# We get the indexes to apply to our variable x to build the samples to
# get an estimation of eps and apply DBSCAN to
eps_idx: np.ndarray = sample_for_every_level_idx(
    valid_dem_values, n_points_to_sample=100_000, n_bins=128)
clust_idx: np.ndarray = sample_for_every_level_idx(
    valid_dem_values, n_points_to_sample=100_000, n_bins=128)

# Now we build the samples of our variable x
x_sample_eps: np.ndarray = valid_dem_values[eps_idx]
x_sample_labels: np.ndarray = valid_dem_values[clust_idx]

print("8. Applying NearestNeighbors algorithm to start estimating eps0")
# k-distance in the *combined* space
nn: NearestNeighbors = NearestNeighbors(n_neighbors=5,
                                        metric="euclidean").fit(x_sample_eps)
print("8.1 We get the distances between points sorted")
dists, _ = nn.kneighbors(x_sample_eps, return_distance=True)
k_dist: np.ndarray = dists[:, -1].astype(np.float32)

percentile_k: int = 95
print("8.2 Getting percentile {0} of distances between points".format(
    percentile_k))
print("We will use it to get esp0_m")
k: float = 0.7
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
            raise ValueError("DEM values are all identical; DBSCAN "
                             "cannot form clusters.")
    else:
        raise ValueError("DEM values are all identical; DBSCAN cannot "
                         "form clusters.")
print("eps_m = {0}".format(eps_m))

print("9. Building the DBSCAN model")
dbscan_mod: DBSCAN = DBSCAN(eps=eps_m, min_samples=50, metric='euclidean',
                            algorithm='ball_tree', n_jobs=-1)
print("9.1 Training the model")
dbscan_mod.fit(x_sample_labels)

print("9.2 Getting the classification labels for each pixel")
# Create full labels array, fill valid positions with cluster labels
labels: np.ndarray = dbscan_mod.labels_


print("9.3 Filtering only points in a cluster as variables to train ")
# Use only labeled (non-noise) subset points
mask_labeled: np.ndarray = labels != -1
x_train: np.ndarray = x_sample_labels[mask_labeled].reshape(-1)
y_train: np.ndarray = labels[mask_labeled]

print("10. Applying KNeighborsClassifier to x_train and y_train")
# We apply KNeighborsClassifier to the results we got from applying
# DBSCAN to the samples we worked out previously
knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=5,
                                                 metric="euclidean",
                                                 n_jobs=1)
knn.fit(x_train.reshape(-1, 1), y_train)

print("10.1 Predicting labels for ALL pixels using KNeighborsClassifier")
# As working with all pixels at the same time will blow up our RAM
# we will apply a divide and conquer strategy, training the points in
# different groups until all the image is classified
print("Applying a divide and conquer strategy, predicting points by "
      "groups")
# Getting all points and flattening them so we can use them to predict
flat_valid_dem: np.ndarray = valid_dem_values.reshape(-1, 1).astype(np.float32)
# Number of points we will predict in each interation
step: int = 500_000
# Variable that will contain the cluster of each point in the image
full_labels_valid: np.ndarray = np.empty(
    flat_valid_dem.shape[0], dtype=np.int32)
i: int = 1
for s in range(0, flat_valid_dem.shape[0], step):
    print("Iteration = {0}".format(i))
    e: int = min(s + step, flat_valid_dem.shape[0])
    print("From s={0} to e={1}".format(s, e))
    full_labels_valid[s:e] = knn.predict(flat_valid_dem[s:e])
    i += 1

# We create an array full of -1 (represents noise)
labels_full = np.full(flat_dem.shape[0], -1)
# In the points with data, we replace all -1 values by their
# corresponding labels
labels_full[flat_mask] = full_labels_valid.astype(np.int32)
n_clusters: int = np.unique(labels_full[labels_full != -1]).size
print("Number of clusters = {0}".format(n_clusters))


print("11. Getting classification labels")
labels: np.ndarray = labels_full

print("11.1 Now we are sorting the labels for the group with the lowest "
      "elevation to the one with the highest")
# We get an array with the list of labels (unique labels)
unique_labels: np.ndarray = np.unique(labels[labels != -1])
# To store the mean of the elevation in a cluster
mean_dems: [float] = []
# Key will be the label and value a tuple with min and max elevation
dem_ranges: dict = {}
# For each label
for lab in unique_labels:
    # We get the mean of the elevation and store it
    mean_dem: float = valid_dem_values[full_labels_valid == lab].mean()
    mean_dems.append((lab, mean_dem))

print("11.2 Sorting the labels by mean elevation")
# Sort clusters by mean elevation ascending
sorted_clusters: [] = sorted(mean_dems, key=lambda x: x[1])
print("11.3 Mapping old labels to the new labels sorted by elevation")
# Create a mapping from old label to new sorted label
old_to_new_label: dict = {old_lab: new_lab for new_lab, (old_lab, _) in
                          enumerate(sorted_clusters)}

print("11.4 Now we replace the old labels for the new sorted labels in "
      "the classified image")
# Remap labels array (keeping nodata as -1)
labels_sorted = np.full(labels.shape, fill_value=-1, dtype=int)
for old_lab, new_lab in old_to_new_label.items():
    labels_sorted[labels == old_lab] = new_lab

print("11.5 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
unique_labels: np.ndarray = np.unique(labels_sorted[labels != -1])
for lab in unique_labels:
    # We turn dem_data into an array and select only the pixels with
    # the corresponding label
    dem_values: np.ndarray = valid_dem_values[labels_sorted[flat_mask] == lab]
    # Now we get the min and max elevations in the cluster and store
    # it in the dictionary
    dem_ranges[lab] = (np.min(dem_values), np.max(dem_values))
print(dem_ranges)

print("12. Post processing data to display it")
# Rebuilding classification map
# We create a map of the shape of our image full of -1 (invalid points)
cluster_map: np.ndarray = np.full(dem_resampled.shape, -1, dtype=np.int16)
# We get valid positions (the ones that are not masked)
valid_positions = np.where(~mask_invalid)
# We add our labels to valid positions in our new cluster map
cluster_map[valid_positions] = labels_sorted[flat_mask]
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("12. Getting DEM ranges")
print("\nElevation range per cluster:")
for cid in range(n_clusters):
    vals = dem_resampled[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .2f} – {vals.max(): .2f}")

# Define discrete colormap and normalization for labels -1 to best k
colors: [str] = colour_scale_array((252, 201, 192), (77, 21, 6), n_clusters)

cmap: ListedColormap = ListedColormap(colors)
cmap.set_bad(color=(0, 0, 0, 0))

plt.figure(figsize=(12, 10))
plt.imshow(mos_resampled, cmap="gray", aspect="auto")
plt.imshow(cluster_map, cmap=cmap, alpha=0.7, aspect="auto")
plt.title(f"Elevation clusters (k={n_clusters}) over Mosaic")

plt.axis('off')
label_names = ["From {0} to {1}".format(elev[0], elev[1]) for elev in
               dem_ranges.values()]
# Create colorbar with ticks
boundaries = np.arange(-0.5, n_clusters + 0.5, 1)
norm = BoundaryNorm(boundaries, ncolors=n_clusters)
cbar = plt.colorbar(ticks=np.arange(0, n_clusters), boundaries=boundaries)
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)

out_png = r".\DBSCAN_mercury_elevation.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig = plt.gcf()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved PNG to: {out_png}")

plt.tight_layout()
plt.show()
