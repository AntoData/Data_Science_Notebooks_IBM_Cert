import random as rd
import numpy as np
import rasterio
from kneed import KneeLocator
from rasterio.warp import reproject, Resampling
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

"""
PROBLEM: We are picking two different GeoTIFF rasters (georeferenced 
TIFF files). These are images with embedded spatial metadata 
(projection/CRS, pixel size, geographic extent, NoData, etc.) 
so any GIS/remote-sensing tool can place them correctly on a map. We 
will classify the information about elevation of the terrain (DEM) 
into clusters and display it over an image of the same area of Pluto's 
surface

The area these files represent is around 100% of Pluto's surface. 
In detail:
Latitude coverage: from −90° to +90° (full global coverage)
Longitude coverage: full 0°–360° domain in simple cylindrical projection 
Pluto’s total surface area ≈ 17.3 million km² (≈ 1.77 × 10⁷km²) 
Pixel size: 300m per pixel (approx. 69 pixels/degree)

The images are:

-   Pluto_NewHorizons_Global_DEM_300m_Jul2017_16bit.tif
    A global digital elevation model (DEM) of Pluto produced by 
    combining stereo observations from NASA’s New Horizons instruments
     (LORRI and MVIC), published by the USGS Astrogeology Science 
     Center, and released on July 14, 2017 
    Elevation is derived from parallax-based stereo photogrammetry and
     represents terrain height relative to Pluto’s reference sphere 
     (radius ≈ 1188.3km) 
    SOURCE:
    https://planetarymaps.usgs.gov/
    mosaic//Pluto_NewHorizons_Global_DEM_300m_Jul2017_16bit.tif

-   Pluto_NewHorizins_Global_Mosaic_300m_Jul2017_8bit.tif
    A global visual image mosaic of Pluto, produced from a combination
    of high-resolution images captured by the Long Range 
    Reconnaissance Imager (LORRI) and the Multispectral Visible 
    Imaging Camera (MVIC) aboard NASA’s New Horizons spacecraft.
    Published by the USGS Astrogeology Science Center, released on 
    July 14, 2017, as part of the global planetary mapping efforts 
    around New Horizons data products 
    Spatial coverage: Entire globe of Pluto, spanning latitudes 
    −90° to +90° and longitudes 0° to 360°, in simple cylindrical 
    (equirectangular) projection
    
    SOURCE:
    https://planetarymaps.usgs.gov/
    mosaic/Pluto_NewHorizons_Global_Mosaic_300m_Jul2017_8bit.tif

We will use image Pluto_NewHorizons_Global_DEM_300m_Jul2017_16bit to 
train a Kmeans  algorithm and classify the different areas of the Pluto
 according to their terrain elevation and them use that information 
 to display this classification over the image 
 Pluto_NewHorizons_Global_Mosaic_300m_Jul2017_8bit. 
 As both refer to the same area
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
dem_path: str = "Pluto_NewHorizons_Global_DEM_300m_Jul2017_16bit.tif"
mos_path: str = "Pluto_NewHorizons_Global_Mosaic_300m_Jul2017_8bit.tif"

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
print("2.1 Reprojectting dataset DEM to dataset mosaic "
      "grid so the area matches")
# Reprojectting dataset DEM to dataset NIR grid so the area matches
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

print("3. Reprojectting Mosaic image to dataset DEM grid so the area matches")
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

print("5. Flatenning our data to use it later to train")
# Use the same mask to index all flattened arrays
flat_dem = dem_resampled.data.ravel()
flat_mask: np.ndarray = ~dem_resampled.mask.ravel()

# Apply mask
valid_dem_values: np.ndarray = flat_dem[flat_mask].reshape(-1, 1)

print("6. Checking we have at least 10 samples")
if valid_dem_values.shape[0] < 10:
    raise ValueError("Too few valid DEM samples to cluster.")

# print("7. Let's apply the elbow method to get the best k for our
# Kmeans model")
# inertias: [] = []
# k_values: [int] = range(3, 11)  # Try k from 3 to 10
#
# for k in k_values:
#     print("7.1 Training the Kmeans algorithm with k={0}".format(k))
#     kmeans_model: KMeans = KMeans(n_clusters=k, random_state=42,
#                                   init="k-means++")
#     kmeans_model.fit(valid_dem_values)
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
best_k: int = 5
print("8. Building the kmeans model")
kmeans_model_train: KMeans = KMeans(n_clusters=best_k, n_init=10,
                                    random_state=0)
print("9. Training the model")
kmeans_model_train.fit(valid_dem_values)

print("10. Getting classification labels")
labels: np.ndarray = kmeans_model_train.labels_

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
    mean_dem: float = valid_dem_values[labels == lab].mean()
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
    dem_values: np.ndarray = valid_dem_values[labels_sorted == lab]
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
cluster_map[valid_positions] = labels_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("12. Getting DEM ranges")
print("\nElevation range per cluster:")
for cid in range(best_k):
    vals = dem_resampled[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .2f} – {vals.max(): .2f}")

# Define discrete colormap and normalization for labels -1 to best k
colors: [str] = ['#{:02x}{:02x}{:02x}'.format(rd.randint(64, 255),
                                              rd.randint(64, 255),
                                              rd.randint(64, 255))
                 for _ in range(0, best_k)]
cmap: ListedColormap = ListedColormap(colors)
cmap.set_bad(color=(0, 0, 0, 0))

plt.figure(figsize=(12, 10))
plt.imshow(mos_resampled, cmap="gray", aspect="auto")
plt.imshow(cluster_map, cmap=cmap, alpha=0.7, aspect="auto")
plt.title(f"Elevation clusters (k={best_k}) over Mosaic")

plt.axis('off')
label_names = ["From {0} to {1}".format(elev[0], elev[1]) for elev in
               dem_ranges.values()]
# Create colorbar with ticks
boundaries = np.arange(-0.5, best_k + 0.5, 1)
norm = BoundaryNorm(boundaries, ncolors=best_k)
cbar = plt.colorbar(ticks=np.arange(0, best_k), boundaries=boundaries)
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)
plt.tight_layout()
plt.show()
