import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from rasterio.warp import reproject, Resampling
from sklearn.cluster import KMeans
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
# Reprojectting dataset DEM to dataset NIR grid so the area matches
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

print("2.2 Reprojectting dataset rou to dataset mosaic "
      "grid so the area matches")
# Reprojectting dataset DEM to dataset NIR grid so the area matches
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

print("2.3 Reprojectting Mosaic image to dataset emm grid so the area matches")
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

print("6. Let's apply the elbow method to get the best k for our Kmeans model")
inertias: [] = []
k_values: [int] = range(3, 11)  # Try k from 3 to 10

for k in k_values:
    print("6.1 Training the Kmeans algorithm with k={0}".format(k))
    kmeans_model: KMeans = KMeans(n_clusters=k, random_state=42,
                                  init="k-means++")
    kmeans_model.fit(x_train_scaled)
    print("6.2 Adding inertia for this k to the array")
    inertias.append(kmeans_model.inertia_)

print("6.3 Plotting the inertias")
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Choosing k")
plt.grid(True)
plt.show()

print("6.4 Applying KneeLocator to get our best k according to the method")
kl: KneeLocator = \
    KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
best_k: int = kl.elbow
print("Best k (elbow found at):", best_k)

print("7. Building the kmeans model")
kmeans_model_train: KMeans = KMeans(n_clusters=best_k, n_init=10,
                                    random_state=0)
print("8. Training the model")
kmeans_model_train.fit(x_train_scaled)

print("9. Getting classification labels")
labels: np.ndarray = kmeans_model_train.labels_

print("10.1 Now we are sorting the labels for the group with the lowest "
      "elevation to the one with the highest")
# We get an array with the list of labels (unique labels)
unique_labels: np.ndarray = np.unique(labels[labels != -1])
# To store the mean of the elevation in a cluster
mean_dems: [float] = []
# Key will be the label and value a tuple with min and max elevation
emm_ranges: dict = {}
rou_ranges: dict = {}
# For each label we get the mean value of emissivity (first value)
means = [(lab, x_train[:, 0][labels == lab].mean()) for lab in
         np.unique(labels)]
# Adding labels to a dictionary to get the translation between new
# sorted labels by mean values and old labels
old_to_new = {old: new for new, (old, _) in enumerate(
    sorted(means, key=lambda t: t[1]))}

print("10.2 Sorting the labels by mean elevation")
# Sort clusters by mean elevation ascending
labels_sorted = np.full_like(labels, -1)
for old, new in old_to_new.items():
    labels_sorted[labels == old] = new
print("10.3 Mapping old labels to the new labels sorted by elevation")
# Create a mapping from old label to new sorted label
cluster_map = np.full(emm_phys.shape, -1, np.int16)
cluster_map.ravel()[valid_flat] = labels_sorted
cluster_map = np.ma.masked_equal(cluster_map, -1)

print("10.4 Now adding the ranges of each sorted label to a dictionary "
      "for later use in legend")
emm_ranges = {}
rou_ranges = {}
for cid in range(best_k):
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
for cid in range(best_k):
    vals = emm_ma[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .3f} – {vals.max(): .3f}")
print("labels:", labels_sorted.size, "valid pixels:", valid_2d.sum())
print("\nRoughness range per cluster:")
for cid in range(best_k):
    vals = rou_ma[cluster_map == cid]
    if vals.size > 0:
        print(f"  Cluster {cid}: {vals.min(): .1f} – {vals.max(): .1f}")


# Define discrete colormap and normalization for labels -1 to best k
colors: [str] = colour_scale_array((252, 201, 192), (77, 21, 6), best_k)

cmap: ListedColormap = ListedColormap(colors)
cmap.set_bad(color=(0, 0, 0, 0))

plt.figure(figsize=(12, 10))
plt.imshow(mos_resampled, cmap="gray", aspect="auto")
plt.imshow(cluster_map, cmap=cmap, alpha=0.7, aspect="auto")
plt.title(f"Emissivity and roughness clusters (k={best_k}) over Mosaic")

plt.axis('off')
label_names = ["Emi: {0}-{1}, Rou: {2}-{3}ºº".
               format(round(emm_ranges[k_][0], 3), round(emm_ranges[k_][1], 3),
                      round(rou_ranges[k_][0], 3), round(rou_ranges[k_][1], 3))
               for k_ in emm_ranges.keys()]
# Create colorbar with ticks
boundaries = np.arange(-0.5, best_k + 0.5, 1)
norm = BoundaryNorm(boundaries, ncolors=best_k)
cbar = plt.colorbar(ticks=np.arange(0, best_k), boundaries=boundaries)
# Set custom tick labels
cbar.ax.set_yticklabels(label_names)
plt.tight_layout()
plt.show()
