import random as rd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from kneed import KneeLocator
from sklearn.cluster import KMeans
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

We will use image Lunar_Clementine_UVVIS_FeO_ClrBinned_70S70N_1km to 
train a Kmeans  algorithm and classify the different areas of the Moon
 according to their FeO concentration and them use that information 
 to display this  classification over the image 
 Lunar_Clementine_NIR_cal_empcor_500m. As both refer to the same area
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
feo_resampled = np.empty(nir_shape, dtype=np.float32)
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
nir_resampled = np.empty(nir_shape, dtype=nir_band.dtype)
reproject(
    source=nir_band,
    destination=nir_resampled,
    src_transform=nir_transform,
    src_crs=nir_crs,
    dst_transform=nir_transform,
    dst_crs=nir_crs,
    resampling=Resampling.bilinear
)
print("3. Getting x and y coordinates")
# Getting x and y coordinates to be included in variable x
rows, cols = np.indices(feo_resampled.shape)
coord_x, coord_y = rasterio.transform.xy(nir_transform,
                                         rows, cols, offset="center")

print("4. Preprocessing data")
print("4.1 Masking invalid data points in FeO")
# Masking invalid values in FeO
mask_invalid = feo_resampled < 0
feo_resampled = np.ma.masked_array(feo_resampled,
                                   mask=mask_invalid)

print("5. Flattening our data to use it later to train")
# Use the same mask to index all flattened arrays
flat_feo = feo_resampled.data.ravel()
flat_mask = ~feo_resampled.mask.ravel()

# Apply mask
valid_feo_values: np.ndarray = flat_feo[flat_mask].reshape(-1, 1)

print("6. Checking we have at least 10 samples")
if valid_feo_values.shape[0] < 10:
    raise ValueError("Too few valid FeO samples to cluster.")

print("7. Let's apply the elbow method to get the best k for our Kmeans model")
inertias: [] = []
k_values: [int] = range(3, 11)  # Try k from 3 to 10

# Apply the elbow method to get the best k
# for k in k_values:
#     print("7.1 Training the Kmeans algorithm with k={0}".format(k))
#     kmeans_model: KMeans = KMeans(n_clusters=k, random_state=42,
#                                   init="k-means++")
#     kmeans_model.fit(valid_feo_values)
#     print("7.2 Adding inertia for this k to the array")
#     inertias.append(kmeans_model.inertia_)
#
# print("7.3 Plotting the inertias")
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, inertias, marker='o')
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Inertia (WCSS)")
# plt.title("Elbow Method for Choosing k")
# plt.grid(True)
# plt.show()
#
# print("7.4 Applying KneeLocator to get our best k according to the method")
# kl: KneeLocator = \
#     KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
# best_k: int = kl.elbow
# print("Best k (elbow found at):", best_k)
# We get as best k = 5
best_k: int = 5

print("8. Building the kmeans model")
kmeans_model_train: KMeans = KMeans(n_clusters=best_k, n_init=10,
                                    random_state=0)
print("9. Training the model")
kmeans_model_train.fit(valid_feo_values)

print("10. Getting classification labels")
labels: np.ndarray = kmeans_model_train.labels_

print("11.1 Now we are sorting the labels for the group with the lowest "
      "FeO to the one with the highest")
# We get an array with the list of labels (unique labels)
unique_labels: np.ndarray = np.unique(labels[labels != -1])
# To store the mean of the elevation in a cluster
mean_feos: [float] = []
# Key will be the label and value a tuple with min and max elevation
feo_ranges: dict = {}
# For each label
for lab in unique_labels:
    # We get the mean of the elevation and store it
    mean_feo: float = valid_feo_values.flatten()[labels == lab].mean()
    mean_feos.append((lab, mean_feo))

print("11.2 Sorting the labels by mean FeO")
# Sort clusters by mean elevation ascending
sorted_clusters: [] = sorted(mean_feos, key=lambda x: x[1])
print("11.3 Mapping old labels to the new labels sorted by FeO")
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
    feo_values: np.ndarray = valid_feo_values.flatten()[labels_sorted == lab]
    # Now we get the min and max elevations in the cluster and store
    # it in the dictionary
    feo_ranges[lab] = (np.min(feo_values), np.max(feo_values))
print(feo_ranges)

print("12. Post processing data to display it")
# Rebuilding classification map
# We create a map of the shape of our image full of -1 (invalid points)
cluster_map: np.ndarray = np.full(feo_resampled.shape, -1, dtype=np.int16)
# We get valid positions (the ones that are not masked)
valid_positions = np.where(~mask_invalid)
# We add our labels to valid positions in our new cluster map
cluster_map[valid_positions] = labels_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("12. Getting FeO ranges")
print("\nFeO range per cluster:")
for cid in range(best_k):
    vals = feo_resampled[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .2f} – {vals.max(): .2f}")

# Define discrete colormap and normalization for labels -1 to best k
colors: [str] = ['#{:02x}{:02x}{:02x}'.format(rd.randint(64, 255),
                                              rd.randint(64, 255),
                                              rd.randint(64, 255))
                 for _ in range(0, best_k)]
cmap: ListedColormap = ListedColormap(colors)
cmap.set_bad(color=(0, 0, 0, 0))
boundaries = np.arange(-0.5, best_k + 0.5, 1)
norm = BoundaryNorm(boundaries, ncolors=best_k)
plt.figure(figsize=(12, 10))
plt.imshow(nir_resampled, cmap="gray", aspect="auto")
plt.imshow(cluster_map, cmap=cmap, alpha=0.3, aspect="auto", norm=norm)
plt.title(f"Clementine FeO Clusters (k={best_k}) over NIR Mosaic")

plt.axis('off')
label_names = ["From {0} to {1}".format(elev[0],
                                        elev[1])
               for elev in
               feo_ranges.values()]
# Create colorbar with ticks
cbar = plt.colorbar(ticks=np.arange(0, best_k), boundaries=boundaries)
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)
plt.tight_layout()
plt.show()
