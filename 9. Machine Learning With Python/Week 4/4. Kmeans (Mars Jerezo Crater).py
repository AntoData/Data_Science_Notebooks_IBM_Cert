import rasterio
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans
from kneed import KneeLocator
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
      "image and th resampled image")
# Stack features (DEM + resampled image), flatten for clustering
x_var: np.ndarray = np.stack([dem_data.flatten(),
                              img_resampled.flatten()], axis=1)

print("5.1 Masking invalid pixels, pixels without data (No Data, Nan...)")
# Mask invalid pixels (NaN or nodata in either input)
valid_mask: np.ndarray = ~np.isnan(x_var).any(axis=1) & \
                         (x_var[:, 0] != dem_src.nodata) \
                         & (x_var[:, 1] != img_src.nodata)

x_valid: np.ndarray = x_var[valid_mask]

# We get best k
# print("6. Let's apply the elbow method to get the best k for our Kmeans model")
# inertias: [] = []
# k_values: [int] = range(1, 10)  # Try k from 1 to 9
#
# for k in k_values:
#     print("6.1 Training the Kmeans algorithm with k={0}".format(k))
#     kmeans_model: KMeans = KMeans(n_clusters=k, random_state=42,
#                                   init="k-means++")
#     kmeans_model.fit(x_valid)
#     print("6.2 Adding inertia for this k to the array")
#     inertias.append(kmeans_model.inertia_)
#
# print("6.3 Plotting the inertias")
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, inertias, marker='o')
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Inertia (WCSS)")
# plt.title("Elbow Method for Choosing k")
# plt.grid(True)
# plt.show()
#
# print("6.4 Applying KneeLocator to get our best k according to the method")
# kl: KneeLocator = \
#     KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
# best_k: int = kl.elbow
# print("Best k (elbow found at):", best_k)
# Best k = 3
best_k: int = 3
print("7. Creating our Kmeans algorithm with best k = {0}".format(best_k))
kmeans_model: KMeans = KMeans(n_clusters=best_k, random_state=42,
                              init="k-means++")
print("8. Training the model with the valid pixels (the ones that have "
      "information)")
kmeans_model.fit(x_valid)
print("9. Getting the classification labels for each pixel")
# Create full labels array, fill valid positions with cluster labels, others -1
labels: np.ndarray = np.full(x_var.shape[0], fill_value=-1, dtype=int)
labels[valid_mask] = kmeans_model.labels_

print("9.1 Reshaping the array to the original shape of the elevation image")
# Reshape to original raster shape
labels_raster: np.ndarray = labels.reshape(dem_data.shape)
print("9.2 Now we are sorting the labels for the group with the lowest "
      "elevation to the one with the highest")
# We get an array with the list of labels (unique labels)
unique_labels: np.ndarray = np.unique(labels[labels != -1])
# To store the mean of the elevation in a cluster
mean_elevations: [float] = []
# Key will be the label and value a tuple with min and max elevation
elev_ranges: dict = {}
# For each label
for lab in unique_labels:
    # We get the mean of the elevation and store it
    mean_elev: float = dem_data.flatten()[labels == lab].mean()
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
labels_sorted = np.full(labels.shape, fill_value=-1, dtype=int)
for old_lab, new_lab in old_to_new_label.items():
    labels_sorted[labels == old_lab] = new_lab


print("9.6 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
unique_labels: np.ndarray = np.unique(labels_sorted[labels != -1])
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

print("10. Plotting results")
# Define discrete colormap and normalization for labels 0 to best k
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
cbar = plt.colorbar(ticks=np.arange(0, best_k))
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)
cbar.set_label('Cluster label')
plt.title('KMeans Classification Overlay on Image')
plt.axis('off')
plt.show()
