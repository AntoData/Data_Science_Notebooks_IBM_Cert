import os
import rasterio
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from hdbscan import HDBSCAN
from rasterio.warp import reproject, Resampling

"""
PROBLEM: We are picking two different GeoTIFF rasters (georeferenced 
TIFF files). These are images with embedded spatial metadata 
(projection/CRS, pixel size, geographic extent, NoData, etc.) 
so any GIS/remote-sensing tool can place them correctly on a map.

The area these files represent is the Jezero Crater in Mars. 
This is a crater on Mars in the Syrtis Major quadrangle, about 45.0 km 
in diameter. Thought to have once been flooded with water, the crater 
contains a fan-delta deposit rich in clays. The lake in the crater was
 present when valley networks were forming on Mars
More information: https://en.wikipedia.org/wiki/Jezero_(crater)

The images are:

-   DEM file (M20_JezeroCrater_CTXDEM_20m.tif)
    Single-band GeoTIFF (float) containing elevation (DEM) values at 20 
    m/pixel, in planetocentric equirectangular (Mars 2000 Sphere) 
    projection, with full georeferencing (CRS, transform, bounds, 
    NoData).
    SOURCE:
    https://astrogeology.usgs.gov/search/map/
    mars_2020_science_investigation_ctx_dem_mosaic

-   CTX orthomosaic (M20_JezeroCrater_CTXortho_mosaic_5m.tif)
    Single-band GeoTIFF (uint16/uint8 grayscale) that is an 
    orthorectified CTX image mosaic at 5 m/pixel, in the same 
    equirectangular Mars projection, georeferenced to align with the 
    DEM.
    SOURCE:
    https://astrogeology.usgs.gov/search/map/
    mars_2020_science_investigation_ctx_dem_mosaic

We will use image M20_JezeroCrater_CTXDEM_20m to train a Kmeans 
algorithm and classify the different areas of the crater according to
their elevation and them use that information to display this 
classification over the image M20_JezeroCrater_CTXortho_mosaic_5m.
As both refer to the same area
"""


def sample_for_every_level_idx(
        var_to_sample: np.ndarray, per_bin: int = 600, n_bins: int = 128,
        cap: int | None = None):
    """
    Returns indexes sampled from equal-width bins over the first
    column of var_to_sample, drawing up to per_bin from each non-empty
    bin to ensure coverage across the variable’s value range.

    :param var_to_sample: Input x variable we want to sample
    :type var_to_sample: np.ndarray
    :param per_bin: Number of points to select per range.By default, 600
    :type per_bin: int
    :param n_bins: Number of equally distributed ranges to build and
    pick per_bin number of points
    :type n_bins: int
    :param cap: Max number of indexes to return to build the sample
    :type cap: int

    :return: Returns indexes sampled from equal-width bins over the
    first column of var_to_sample, drawing up to per_bin from each
     non-empty bin
    :rtype: np.ndarray
    """
    var_1d: np.ndarray = var_to_sample[:, 0]  # 1-D
    # Getting max and min values of variable
    x_min: float
    x_max: float
    x_min, x_max = float(np.min(var_1d)), float(np.max(var_1d))
    # We get n_bins + 1 evenly spaced values in our variable which will
    # become edges of ranges of our classification
    edges: np.ndarray = np.linspace(x_min, x_max, n_bins + 1)

    # We get the indexes of the points with lowest and highest value
    i_min: int = int(np.argmin(var_to_sample[:, 0]))
    i_max: int = int(np.argmax(var_to_sample[:, 0]))
    # Initially we add indexes for points with min and max values
    keep: [] = [i_min, i_max]
    # Numpy random generator
    rng: np.random.Generator = np.random.default_rng(42)
    for b in range(n_bins):
        # We get only the points in our x variable that belong to this
        # range
        sel: np.ndarray = np.where((var_1d >= edges[b])
                                   & (var_1d < edges[b + 1]))[0]
        # If any point belonged to this range
        if sel.size:
            # n_samples reduces the number of points to take for our
            # sample (the min between the number of points per bin we
            # intended or the size of the points in our variable that
            # belong to this range)
            n_samples = min(per_bin, sel.size)
            # We pick n_sample points from our variable x that belong
            # to this range
            keep.append(rng.choice(sel, n_samples, replace=False))
    # This take keep which is an array of arrays of indexes and flattens
    # it to 1 dimension, removing possible repeated indexes
    points_indexes: np.ndarray = np.unique(np.concatenate(
        [np.atleast_1d(k_) for k_ in keep]))
    # If we set a size to cap our sample and this size is smaller than
    # our current sample
    if cap is not None and points_indexes.size > cap:
        # We pick cap number of points of our indexes
        points_indexes = rng.choice(points_indexes, size=cap, replace=False)
    # We return the indexes of the points we want to use as samples
    return points_indexes


# Paths to your files
dem_path: str = 'M20_JezeroCrater_CTXDEM_20m.tif'
img_path: str = 'M20_JezeroCrater_CTXortho_mosaic_5m.tif'

print("1. Open the Tif files")
# Open datasets and print metadata
with rasterio.open(dem_path) as dem_src, rasterio.open(img_path) as img_src:
    print("DEM CRS:", dem_src.crs)
    print("Image CRS:", img_src.crs)
    print("DEM bounds:", dem_src.bounds)
    print("Image bounds:", img_src.bounds)
    print("DEM resolution:", dem_src.res)
    print("Image resolution:", img_src.res)
    print("")
    print("2. Let's check their boundaries are very close")
    assert abs(dem_src.bounds.left - img_src.bounds.left) < 200
    assert abs(dem_src.bounds.bottom - img_src.bounds.bottom) < 200
    assert abs(dem_src.bounds.right - img_src.bounds.right) < 200
    assert abs(dem_src.bounds.top - img_src.bounds.top) < 200
    print("2.1 Images match enough for this problem, boundaries are "
          "similar enough")
    print("")

    print("3. Reading the elevation file (it has a single band, it is "
          "gray scale")
    # Read DEM data (single band) as it is gray scale
    dem_data: np.ndarray = dem_src.read(1)

    print("4. Sampling the image M20_JezeroCrater_CTXortho_mosaic_5m "
          "so it has the same size as M20_JezeroCrater_CTXDEM_20m")
    # We create an empty array for resampled image data (match DEM grid)
    # this will be where we will allocate the projection
    img_resampled: np.ndarray = np.empty((dem_src.height, dem_src.width),
                                         dtype=img_src.dtypes[0])

    # Resample image to DEM grid (match resolution and CRS)
    reproject(
        source=img_src.read(1),
        destination=img_resampled,
        src_transform=img_src.transform,
        src_crs=img_src.crs,
        dst_transform=dem_src.transform,
        dst_crs=dem_src.crs,
        resampling=Resampling.bilinear
    )

print("5. Preprocessing, stacking the features of the elevation "
      "image and the resampled image")
# Stack features (DEM + resampled image), flatten for clustering
x_var: np.ndarray = np.stack([dem_data.flatten()], axis=1)

print("5.1 Masking invalid pixels, pixels without data (No Data, Nan...)")
# Mask invalid pixels (NaN or nodata in either input)
valid_mask: np.ndarray = ~np.isnan(x_var).any(axis=1) & \
                         (x_var[:, 0] != dem_src.nodata)

x_valid: np.ndarray = x_var[valid_mask]

print("5.2 Sampling our variable x")
# HDBSCAN is very costly for this amount of points so we will select a
# number of random points to train our HDBSCAN algorithm. These points
# will be the seed for a KNearestNeighbor algorithm which will apply
# the last part of the classification

# However we will use the method defined and the beginning of the file
# to make sure our sample's points are representative of our variable

# We get the number of elements of our variable x
n_points: int = x_valid.shape[0]
# We set the size of our samples
eps_subset_size = min(800_000, n_points)  # for eps estimation
clust_subset_size = min(800_000, n_points)  # for HDBSCAN itself

# We get the indexes to apply to our variable x to build the samples to
# get an estimation of eps and apply HDBSCAN to
eps_idx: np.ndarray = sample_for_every_level_idx(
    x_valid, per_bin=400, n_bins=128, cap=min(50_000, n_points))
clust_idx: np.ndarray = sample_for_every_level_idx(
    x_valid, per_bin=400, n_bins=128, cap=min(100_000, n_points))

# Now we build the samples of our variable x
x_sample_eps = x_valid[eps_idx]
x_sample_labels = x_valid[clust_idx]

print("6. Apply NearestNeighbors algorithm to start estimating eps0")
# k-distance in the *combined* space
nn = NearestNeighbors(n_neighbors=5,
                      metric="euclidean").fit(x_sample_eps)
print("6.1 We get the distances between points sorted")
dists, _ = nn.kneighbors(x_sample_eps, return_distance=True)
kdist: np.ndarray = dists[:, -1].astype(np.float32)

print("6.2 Getting percentile 90 of distances between points")
eps0_m = np.percentile(kdist, 90)
k: float = 3.2
print("7. Creating the HDBSCAN algorithm with k={0}".format(k))
# Apply K-Means
print("7.1 Multiplier k = {0}".format(k))
eps_m = float(eps0_m * k)
print("7.2 Building the HDBSCAN object")
hdbscan_mod = HDBSCAN(min_samples=10, metric='euclidean',
                      min_cluster_size=20,
                      cluster_selection_method="eom",
                      # merges → fewer gaps
                      cluster_selection_epsilon=eps_m,
                      prediction_data=True
                      )
print("7.3 Training the model")
hdbscan_mod.fit(x_sample_labels)

print("8. Getting the classification labels for each pixel")
# Create full labels array, fill valid positions with cluster labels
labels: np.ndarray = hdbscan_mod.labels_

print("8.1 Filtering only points in a cluster as variables to train ")
# Use only labeled (non-noise) subset points
mask_labeled = labels != -1
x_train = x_sample_labels[mask_labeled].reshape(-1)  # 1-D feature
y_train = labels[mask_labeled]

print("9. Applying KNeighborsClassifier to x_train and y_train")
# We apply KNeighborsClassifier to the results we got from applying
# HDBSCAN to the samples we worked out previously
knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=5,
                                                 metric="euclidean",
                                                 n_jobs=1)
knn.fit(x_train.reshape(-1, 1), y_train)
full_labels_valid = knn.predict(x_valid)

# We create an array full of -1 (represents noise)
labels_full = np.full(x_var.shape[0], fill_value=-1, dtype=np.int32)
# In the points with data, we replace all -1 values by their
# corresponding labels
labels_full[valid_mask] = full_labels_valid
n_clusters: int = np.unique(labels_full[labels_full != -1]).size
print("Number of clusters = {0}".format(n_clusters))

print("9.1 Reshaping the array to the original shape of the elevation image")
# Reshape to original raster shape
labels_raster: np.ndarray = labels_full.reshape(dem_data.shape)
print("9.2 Now we are sorting the labels for the group with the lowest "
      "elevation to the one with the highest")
# We get an array with the list of labels (unique labels)
unique_labels: np.ndarray = np.unique(labels_full[labels_full != -1])
# To store the mean of the elevation in a cluster
mean_elevations: [float] = []
# Key will be the label and value a tuple with min and max elevation
elev_ranges: dict = {}
# For each label
for lab in unique_labels:
    # We get the mean of the elevation and store it
    mean_elev: float = dem_data.flatten()[labels_full == lab].mean()
    mean_elevations.append((lab, mean_elev))

print("9.3 Sorting the labels by mean elevation")
# Sort clusters by mean elevation ascending
sorted_clusters: [] = sorted(mean_elevations, key=lambda x: x[1])
print("9.4 Mapping old labels to the new labels sorted by elevation")
# Create a mapping from old label to new sorted label
old_to_new_label = {old_lab: new_lab for new_lab, (old_lab, _) in
                    enumerate(sorted_clusters)}

print("9.5 Now we replace the old labels for the new sorted labels in "
      "the classified image")
# Remap labels array (keeping nodata as -1)
labels_sorted = np.full(labels_full.shape, fill_value=-1, dtype=int)
for old_lab, new_lab in old_to_new_label.items():
    labels_sorted[labels_full == old_lab] = new_lab

print("9.6 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
unique_labels: np.ndarray = np.unique(labels_sorted[labels_full != -1])
for lab in unique_labels:
    # We turn dem_data into an array and select only the pixels with
    # the corresponding label
    elev_values: np.ndarray = dem_data.flatten()[labels_sorted == lab]
    # Now we get the min and max elevations in the cluster and store
    # it in the dictionary
    elev_ranges[lab] = (np.min(elev_values), np.max(elev_values))
print(elev_ranges)

print("9.7 Reshaping the image classified in our sorted clusters "
      "to match our original elevation image")
# Reshaping the image classified in our sorted clusters to match
# elevation image
labels_raster_sorted: np.ndarray = labels_sorted.reshape(dem_data.shape)
best_k = int(np.unique(labels_sorted[labels_sorted != -1]).size)
print("10. Plotting results")
# Define discrete colo1rmap and normalization for labels 0 to best k
colors: [str] = ['#{:02x}{:02x}{:02x}'.format(rd.randint(64, 255),
                                              rd.randint(64, 255),
                                              rd.randint(64, 255))
                 for _ in range(0, best_k)]
cmap: ListedColormap = ListedColormap(colors)
boundaries: np.ndarray = np.arange(-0.5, best_k + .5, 1)
norm: BoundaryNorm = BoundaryNorm(boundaries, cmap.N)

# Plot the results
plt.figure(figsize=(10, 10))
plt.imshow(img_resampled, cmap='gray', alpha=0.7)
plt.imshow(labels_raster_sorted, cmap=cmap, norm=norm,
           alpha=0.4)  # overlay with discrete colors
label_names = ["From {0} to {1}".format(elev[0], elev[1]) for elev in
               elev_ranges.values()]
# Create colorbar with ticks
cbar = plt.colorbar(ticks=np.arange(0, len(label_names)))
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)
cbar.set_label('Cluster label')
plt.title('HDBSCAN Classification Overlay on Image')
plt.axis('off')

out_png = r".\mars_jerezo_hdbscan.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig = plt.gcf()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved PNG to: {out_png}")
dem_flat = dem_data.flatten()
global_min = float(dem_flat[valid_mask].min())
global_max = float(dem_flat[valid_mask].max())
map_min = float(dem_flat[labels_sorted != -1].min())
map_max = float(dem_flat[labels_sorted != -1].max())
print(f"[CHECK] DEM min/max: {global_min: .3f} / {global_max: .3f}")
print(f"[CHECK] Map min/max: {map_min: .3f} / {map_max: .3f}")
plt.show()
"""
CONCLUSIONS: HDBSCAN consumes a lot of RAM memory if the sample has 
considerable size. Several times while trying to solve this problem with 
other parameters crashed my machine. 
In the end, the solution was to sample our variable and use HDBSCAN with
a sample and then use those labels as seeds for KNeighborsClassifier
Also, we had to built a custom made method to sample our variable so
it would take points from all different ranges of the variable
"""