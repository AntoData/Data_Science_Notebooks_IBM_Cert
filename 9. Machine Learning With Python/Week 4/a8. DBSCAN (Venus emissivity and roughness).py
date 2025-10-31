import os
import math
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler
from rasterio.warp import reproject, Resampling
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
and Venus_Magellan_MeterScaleSlope_Global_4641m.tif to train a KMeans
algorithm and classify the different areas of Venus according to their
microwave emissivity and meter-scale surface roughness and then use that
information to display this classification over the image
Venus_Magellan_C3-MDIR_Colorized_Global_Mosaic_4641m.tif. As both refer
to the same area
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
    emm_bounds, rou_bounds, mos_bounds = emm_src.bounds, rou_src.bounds,\
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
x_train_scaled: np.ndarray = StandardScaler().fit_transform(x_train)

print("7. Sampling our variable x")
# DBSCAN is very costly for this amount of points so we will select a
# number of random points to train our DBSCAN algorithm. These points
# will be the seed for a KNearestNeighbor algorithm which will apply
# the last part of the classification

# However we will use the method defined and the beginning of the file
# to make sure our sample's points are representative of our variable

# We get the number of elements of our variable x
n_points: int = x_train_scaled.shape[0]
# We set the size of our samples
eps_subset_size: int = min(2_000_000, n_points)  # for eps estimation
clust_subset_size: int = min(2_000_000, n_points)  # for DBSCAN itself

# We get the indexes to apply to our variable x to build the samples to
# get an estimation of eps and apply DBSCAN to
eps_idx: np.ndarray = sample_for_every_level_idx(
    x_train_scaled, n_points_to_sample=100_000, n_bins=128)
clust_idx: np.ndarray = sample_for_every_level_idx(
    x_train_scaled, n_points_to_sample=100_000, n_bins=128)

# Now we build the samples of our variable x
x_sample_eps: np.ndarray = x_train_scaled[eps_idx]
x_sample_labels: np.ndarray = x_train_scaled[clust_idx]

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
mask_labeled = labels != -1
x_train_knn = x_sample_labels[mask_labeled]  # shape: (n_samples, 2)
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

# --- Sort cluster IDs by emissivity mean, using FULL KNN predictions ---
valid_emm = emm_flat[valid_flat]  # emissivity for valid pixels only
labs = np.unique(full_labels_valid[full_labels_valid != -1])

# mean emissivity per predicted cluster (same valid subset length)
means = [(lab, valid_emm[full_labels_valid == lab].mean()) for lab in labs]
old_to_new = {old: new for new, (old, _) in enumerate(sorted(means, key=lambda t: t[1]))}

# Remap predicted labels to the sorted order
full_labels_valid_sorted = np.vectorize(old_to_new.get)(full_labels_valid)

print("10.2 Sorting the labels by mean elevation")
# Build 2D cluster map for the entire raster (only valid pixels get labels)
cluster_map = np.full(emm_resampled.shape, -1, dtype=np.int16)
cluster_map.ravel()[valid_flat] = full_labels_valid_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("10.4 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
emm_ranges = {}
rou_ranges = {}
for cid in range(len(labs)):
    e_vals = emm_ma[cluster_map == cid]
    r_vals = rou_ma[cluster_map == cid]
    if e_vals.size:
        emm_ranges[cid] = (float(e_vals.min()), float(e_vals.max()))
        rou_ranges[cid] = (float(r_vals.min()), float(r_vals.max()))

print("10.2 Sorting the labels by mean elevation")
# Sort clusters by mean elevation ascending
labels_sorted = np.full_like(labels, -1)
for old, new in old_to_new.items():
    labels_sorted[labels == old] = new

print("10.3 Mapping old labels to the new labels sorted by elevation")
# Create a mapping from old label to new sorted label
cluster_map = np.full(emm_phys.shape, -1, np.int16)
labels_sorted = full_labels_valid_sorted  # length == valid_flat.sum()
cluster_map.ravel()[valid_flat] = labels_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("10.4 Now adding the ranges of each sorted label to a dictionary "
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
valid_positions = np.where(valid_2d)   # ← use the SAME mask you trained with
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
