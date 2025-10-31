import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colors as mcolors
from skimage.io import imread
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter, binary_opening
from scipy.ndimage import label
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

"""
SOURCE: Image comes from: 
https://skyserver.sdss.org/dr19/VisualTools/navi

The goal is to classify the pixels on the image to identify the
celestial objects in the object and isolate them from the dark
background and classify those elements depending on their intensity
"""


def distinct_palette(n: int, sat: float = 0.85, val: float = 0.98,
                     hue_offset: float = 0.10) -> np.ndarray:
    """
    Returns a palette of n distinct colours and each colour is separated
    of the following one so they don't merge
    :param n: Number of colour in the palette
    :type n: int
    :param sat: Saturation, how “vivid” the colors are in HSV space.
    Higher = purer, more intense colors; lower = more pastel/washed-out.
    :type sat: float
    :param val: Brightness in HSV.
    Higher = brighter colors (closer to white); lower = darker colors.
    :param hue_offset: Where on the hue wheel the sequence starts.
    It rotates all hues by this fraction of the full circle (
    0.0–1.0, where 1 wraps around). Use it to avoid starting near a
    troublesome color
    :return: Array with n distinct and separated colours
    :rtype: np.ndarray
    """
    # evenly spaced hues
    h: np.ndarray = (np.arange(n) / n + hue_offset) % 1.0
    # golden-angle permutation to maximize separation of consecutive indices
    phi: float = (np.sqrt(5) - 1) / 2  # ≈0.618
    order: np.ndarray = np.mod(np.arange(n) * phi, 1).argsort()
    h = h[order]
    s: np.ndarray = np.full(n, sat)
    v: np.ndarray = np.full(n, val)
    # HSV -> RGB
    rgb_array: np.ndarray = matplotlib.colors.hsv_to_rgb(np.c_[h, s, v])
    return rgb_array


print("1. Opening the image")
# We use imread (image read) from skimage
image: np.ndarray = imread('skyserver.png')

print("2. Preprocessing the image")
print("2.1 We reshape the image if it has 4 channels RGB and brightness")
if image.shape[-1] == 4:
    print("It has 4 channels")
    image = image[:, :, :3]
    print("Reshaped to 3")

print("2.2 Modifying type to float")
img: np.ndarray = image.astype(np.float32, copy=False)
print("2.3 Normalizing the image")
# Normalize by dtype
if image.dtype == np.uint8:
    print("- Original image was uint8")
    img /= 255.0
elif image.dtype == np.uint16:
    print("- Original image was uint16")
    img /= 65535.0
else:
    print("- Original image was not uint8 nor uint 16")
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)

print("2.4 Applying Gaussian Filter")
# The Gaussian blur removes slow variations so DBSCAN sees compact
# star blobs instead of a drifting, noisy background that would
# otherwise merge into one huge cluster.
blur: np.ndarray = gaussian_filter(
    img, sigma=(10.46, 10.46, 0))  # σ 10–15 works
highpass: np.ndarray = np.clip(img - blur, 0.0, 1.0)
hp_gray: np.ndarray = (0.2126 * highpass[..., 0] +
                       0.7152 * highpass[..., 1] +
                       0.0722 * highpass[..., 2]).astype(np.float32)

print("2.5 Turning the image into a pixels array")
# Reshape for clustering
# Flattening a color image from shape (H, W, 3) into a 2-D array of
# shape (H*W, 3):
# Each row = one pixel,
# 3 columns = R, G, B values,
# The -1 tells NumPy to infer H*W automatically.
print("Flattening image and getting their RGB colour values")
pixels: np.ndarray = image.reshape(-1, 3)
print("Getting Height and Width")
H, W = image.shape[:2]
print("Height = {0}, Width = {1}".format(H, W))
# Building a two 2-D arrays that give you the row and column index of
# every pixel
yy, xx = np.mgrid[0:H, 0:W]
# Normalizing x and y coordinates
x_norm: np.ndarray = (xx / (W - 1)).astype(np.float32).reshape(-1, 1)
y_norm: np.ndarray = (yy / (H - 1)).astype(np.float32).reshape(-1, 1)
# Normalizing colour features
rgb: np.ndarray = (pixels / 255.0).astype(np.float32)
# Weights for x and y and RGB colour
w_xy, w_col = 1.00, 1.00
print("Building our weighted data matrix of features for X")
pixels_x: np.ndarray = np.hstack([x_norm * w_xy, y_norm * w_xy,
                                  rgb * w_col]).astype(np.float32)

print("2.6 Removing the dark background")
# We need to do this in order for the magnitude of operations to be
# doable in a regular laptop. We are not interested in classifying the
# dark background so we can remove it
# keep your pixels_x build exactly as in your file
print("Getting rgb of pixels normalized again")
rgb01: np.ndarray = (pixels / 255.0).astype(np.float32)

print("2.7 Creating foreground mask (which pixels we keep for clustering)")
# Use the high-pass luminance (stars pop, background suppressed) with
# shape (H, W), values ~[0,1].
hp: np.ndarray = hp_gray
# Floor is the 5% darkest pixels
thr_q: float = float(np.quantile(hp, 0.05))
# Adaptive mean+σ cutoff; higher when the image is noisier/brighter.
thr_s: float = float(hp.mean() + 0.30 * hp.std())
# Take the stricter of the two thresholds, pixels above it are kep
keep_mask: np.ndarray = (hp >= max(thr_q, thr_s))
# To remove tiny specks and break thin bridges between pixels so the
# background doesn't percolate into one giant cluster
keep_mask = binary_opening(keep_mask, structure=np.ones((3, 3), bool),
                           iterations=1)

print("2.8 Preprocessing to remove tiny connected blogs from binary mask")
# Labels each connected component in keep_mask (background is 0).
# With no structure, this is 4-connected in 2D.
lbl, _ = label(keep_mask)
# Counts how many pixels each label has: sizes[k] = area of component k
sizes: np.ndarray = np.bincount(lbl.ravel())
# Mark components with area < 6 pixels to be removed
rm: np.ndarray = sizes < 6
# We make sure the background label is not removed
rm[0] = False
# For every pixel, we look up its component label lbl[i,j];
# if that label is marked in rm, we set the pixel to False
# (delete the tiny blob)
keep_mask[rm[lbl]] = False

# We flatten our mask
keep: np.ndarray = keep_mask.reshape(-1)
# We get the array we will work with (making it type float) and keeping
# only the pixels that were not ruled out during our preprocessing
pixels_x_keep: np.ndarray = pixels_x[keep].astype(np.float32, copy=False)

print("3.1 Setting the number of minimum sample per cluster")
min_samples: int = 4
print("min_samples = {0}".format(min_samples))

print("3.2 Getting epsilon 0 to apply later in DBSCAN model")
n_samples: int = 20_000
print("Max number of samples to get from image = {0}".format(n_samples))
print("Getting the number of samples to pick")
m = min(n_samples, pixels_x_keep.shape[0])
print("Final number of samples = {0}".format(m))
print("Sampling image")
idx: np.ndarray = np.random.choice(pixels_x_keep.shape[0], m, replace=False)
print("Building neighbor index on sampled points to query distances to "
      "nearby points efficiently")
nn: NearestNeighbors = NearestNeighbors(
    n_neighbors=min_samples, algorithm="ball_tree", metric="euclidean").fit(
    pixels_x_keep[idx])
print("Building k-distance vector")
kdist: np.ndarray = np.sort(nn.kneighbors(pixels_x_keep[idx])[0][:, -1])
distance_percentile: float = 90.0
print("Picking as epsilon 0 the distance in percentile = {0}".format(
    distance_percentile))
eps = float(np.percentile(kdist, distance_percentile))
eps = min(eps, 0.065)
# Epsilon-0 multiplier
k: float = 1.0
print("Epsilon multiplier k = {0}, so we are applying epsilon = "
      "{1} * {0}".format(k, eps))
print("4. Creating the DBSCAN algorithm with k={0}".format(k))
# Apply DBSCAN
eps_m: float = float(eps * k)
print("Epsilon = {0}".format(eps_m))
print("4.1 Building the DBSCAN object")
dbscan_mod: DBSCAN = DBSCAN(eps=eps_m, min_samples=min_samples,
                            metric="euclidean", algorithm="ball_tree",
                            leaf_size=40, n_jobs=1)
print("4.2 Training the model")
dbscan_mod.fit(pixels_x_keep)
print("4.3 Getting labels for each point")
labels_keep: np.ndarray = dbscan_mod.labels_
print("4.4 Building labels matrix with the same size as our image adding"
      "points we got labels for and adding -1 to points we removed as "
      "they were dark background (-1 is are non-clustered points)")
# Building a matrix of same size as our image and filling it with -1
labels: np.ndarray = np.full(pixels_x.shape[0], -1, dtype=np.int32)
# Adding our clusters to elements in the positions of the points we
# used to train the model
labels[np.where(keep)[0]] = labels_keep

print("5.1 Getting cluster centers")
# Valid points are points in any cluster but -1 (which is the label for
# points without a cluster)
valid: np.ndarray = labels != -1
# If there are valid points, any point that got a cluster
if np.any(valid):
    # Getting clusters ids
    cluster_ids: np.ndarray = np.unique(labels[valid])
    # Working out the center of every cluster
    centers: np.ndarray = np.vstack(
        [pixels[labels == cid].mean(axis=0) for cid in
         cluster_ids])  # shape (K, 3)
else:
    # Otherwise, there are no clusters and centers are 0,0,0
    cluster_ids = np.array([], dtype=int)
    centers = np.empty((0, 3), dtype=float)

# Getting noise or points without clusters in the points we used to
# train the model, the points whose brightness fulfilled the threshold
mask_noise: np.ndarray = keep & (labels == -1)
# Getting valid points that belong to a cluster
mask_core: np.ndarray = keep & (labels != -1)

print("6. Post-processing: Background will stay background (black)"
      " even if DBSCAN accidentally assigned a label to a big dark region")
# Mask the darkest cluster only if it's large (avoid nuking faint stars)
# We reshape again our labels to the size of our original image
seg: np.ndarray = labels.reshape(H, W)
# We create a boolean mask the same shape as labels (1-D, length H*W)
# where if True, pixels belong to a real cluster and False, pixels are
# noise/background. As DBSCAN uses −1 for noise, using >=0 creates a
# mask only for valid points
valid: np.ndarray = labels >= 0
# If at least one element in our mask is True, which means at least
# one point in the image is in a cluster
if valid.any():
    # We get clusters ids that are not background
    cl: np.ndarray = np.unique(labels[valid])
    # We get the mean of RGB for each cluster
    means: np.ndarray = \
        np.vstack([pixels[labels == c].mean(axis=0) for c in cl])
    # We create a brightness proxy per cluster as the Euclidean norm
    # of its mean RGB
    bright: np.ndarray = np.linalg.norm(means, axis=1)
    # We count how many pixels each cluster has
    counts: np.ndarray = np.array([(labels == c).sum() for c in cl])
    # We find the index of the darkest cluster (smallest brightness)
    dark_idx: int = int(np.argmin(bright))
    # We get the darkest cluster's ID
    dark_c: int = cl[dark_idx]
    # We create a copy of the matrix that contains the cluster ID of
    # our pixels
    masked: np.ndarray = seg.copy()
    # We only mask if that darkest cluster is large (likely sky), not a
    # tiny faint star
    if counts[dark_idx] > 0.45 * counts.sum():
        masked[seg == dark_c] = -1
else:
    masked = seg

print("7–8. Plotting original + clusters with legend in one figure")
labels_img: np.ndarray = \
    masked  # after your darkest-cluster masking; shape (H, W)

# Collect real cluster ids (exclude -1 background)
clusters: np.ndarray = np.unique(labels_img[labels_img != -1])
n_clusters: int = len(clusters)
print("Number of clusters: {0}".format(n_clusters))
print("Creating figures for both images, original and segmented")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

print("7. Plotting original image")
# Left: original image
ax1.imshow(image)
ax1.set_title("Original")
ax1.axis("off")
print("8. Plotting segmented image")
# Right: clusters with discrete colors + colorbar legend
ax2.set_facecolor("black")
# If there were no clusters, we just print a message in the window
if n_clusters == 0:
    ax2.text(0.5, 0.5, "No clusters", ha="center", va="center", color="w",
             fontsize=14)
    ax2.axis("off")
else:
    n_colors: int = \
        n_clusters  # clusters = np.unique(labels_img[labels_img != -1])

    # mask background so it renders black
    lab: np.ma.masked_array = \
        np.ma.masked_where(labels_img < 0, labels_img.astype(int))

    # distinct palette (you already defined distinct_palette)
    colors: np.ndarray = distinct_palette(n_colors)  # (K, 3) in 0..1
    cmap: ListedColormap = ListedColormap(colors)
    try:
        cmap = cmap.with_extremes(bad=(0, 0, 0, 1))  # masked -> black
    except AttributeError:
        cmap.set_bad(color=(0, 0, 0, 1))

    # discrete boundaries centered on each actual cluster id
    bounds: np.ndarray = np.r_[clusters - 0.5, clusters[-1] + 0.5]
    norm: matplotlib.colors.BoundaryNorm = \
        mcolors.BoundaryNorm(bounds, n_colors)

    # draw segmented image
    im = ax2.imshow(lab, cmap=cmap, norm=norm, interpolation="nearest")
    ax2.set_title("Clusters (DBSCAN-style)")
    ax2.set_facecolor("black")
    ax2.axis("off")

    # save exactly-as-shown PNG (background stays black)
    rgba: np.ndarray = cmap(norm(lab))
    rgba = np.ma.filled(rgba, (0, 0, 0, 1))
    rgb8: np.ndarray = (rgba[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb8).save("skyserver_dbscan.png")

    # swatch-grid legend (replaces colorbar; scales to ~400 clusters)
    divider = make_axes_locatable(ax2)
    ax_sw = divider.append_axes("right", size="22%", pad=0.10)
    ax_sw.set_title(f"{n_colors} clusters", fontsize=9)
    ax_sw.set_xticks([])
    ax_sw.set_yticks([])

    n_cols: int = min(24, n_colors)  # 20–28 looks good; 24 default
    n_rows: int = int(np.ceil(n_colors / n_cols))
    swatch: np.ndarray = np.zeros((n_rows, n_cols, 3),
                                  dtype=float)  # empty cells = black

    for i in range(n_colors):
        r, c = divmod(i, n_cols)
        swatch[r, c, :] = colors[i]  # colors[i] corresp to clusters[i]

    ax_sw.imshow(swatch, interpolation="nearest", origin="upper")
    ax_sw.set_xlim(0, n_cols)
    ax_sw.set_ylim(n_rows, 0)

    # sparse labels so it stays readable (≈<=60 labels total)
    label_every: int = max(1, n_colors // 60)
    for i in range(0, n_colors, label_every):
        r, c = divmod(i, n_cols)
        txt_color: str = "k" if np.mean(colors[i]) > 0.6 else "w"
        ax_sw.text(c + 0.5, r + 0.5, str(clusters[i]),
                   ha="center", va="center", fontsize=6, color=txt_color)

plt.tight_layout()
plt.show()
"""
CONCLUSIONS: DBSCAN consumes a lot of RAM memory if the sample has 
considerable size. Several times while trying to solve this problem with 
other parameters crashed my machine. It took a lot of research and 
trial and error to get this combination of optimizations and settings 
to make it work. We had to filter the dark background, finetune the 
settings...
Also controlling the number of clusters is extremely difficult
"""