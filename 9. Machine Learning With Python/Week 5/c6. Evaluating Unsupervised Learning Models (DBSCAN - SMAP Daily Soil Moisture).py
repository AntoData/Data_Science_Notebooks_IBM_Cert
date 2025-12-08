import math
import folium
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, \
    davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from pyproj import Transformer

"""
SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture, Version 9
SOURCE: https://nsidc.org/data/spl3smp/versions/9
Soil Moisture Active Passive by NASA
This dataset comes from NASA’s SMAP (Soil Moisture Active Passive) 
satellite, which is designed to measure global soil moisture.

Columns in variable X:
Soil moisture: Volumetric measurements (in m³/m³) of surface soil 
moisture (~top 5 cm).
Brightness temperature: Thermal microwave radiation emitted from the 
Earth’s surface, measured by SMAP’s radiometer.
Geolocation: Latitude and longitude for each grid cell (36 × 36 km).

The problem is to classify the different types of soil and plotting 
the classification in a world map using Folium

NOTE: The files were originally in format .h5 but we converted them to
csv using ChatGPT
"""


def plot_silhouette_scores(silhouette_avg: float,
                           sample_silhouette_values: [float],
                           labels_: [int], performance_summary: str,
                           title_suffix: str = '') -> None:
    """
    Displays the silhouette scores of every point in a model by grouped
    label and compares it to average silhouette score

    :param silhouette_avg: Avg Silhouette score of the model
    :type silhouette_avg: float
    :param sample_silhouette_values: Silhouette scores of every point
    in the variable (these are the points we plot)
    :type sample_silhouette_values: [float]
    :param labels_: Labels assigned to each point in variable X
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
    unique_labels: [int] = np.unique(labels_)
    colormap: plt.Colormap = plt.get_cmap("tab10")
    color_dict: {int: float} = {
        label: colormap(float(label) / len(unique_labels))
        for label in unique_labels}
    y_lower: int = 10
    for ul in unique_labels:
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
    Davies Bouldin indexes across k
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
    :param davies_bouldin_indexes_: Davies Bouldin indexes for the
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


def plot_voronoi_2d(var_x: pd.DataFrame, var_y: np.ndarray,
                    centroids_: np.ndarray, scaler: StandardScaler) -> None:
    """
    Given a variable x that contains columns x and y in metres
    with corresponding classification labels and an array of centroids
    of those clusters, we transform them to a Euclidean space if needed
     and plot the points and their voronoi 2D diagrams

    :param var_x: Variable X of the model
    :type var_x: pd.DataFrame
    :param var_y: Labels predicted by the model for variable X
    :type var_y: np.ndarray
    :param centroids_: Centroids of the model
    :type centroids_: np.ndarray
    :param scaler: Standard Scaler model used to scale variable
    :type scaler: StandardScaler

    :return: None
    """
    print("Getting the original value of the centroids, as variable "
          "x was scaled, we need to unscale them")
    # 1️⃣ Descale centroids back to original units
    descaled_centroids_: np.ndarray = scaler.inverse_transform(centroids_)

    all_scaled_cols = [c for c in var_x.columns if c.endswith('_scaled')]
    x_index: int = all_scaled_cols.index("x_m_scaled")
    y_index: int = all_scaled_cols.index("y_m_scaled")

    # 3) build the k×2 array with ONLY (x,y) in meters
    vor_centroids: np.ndarray = descaled_centroids_[:, [x_index, y_index]]
    vor_centroids: np.ndarray = np.unique(vor_centroids, axis=0)

    print("Creating the voronoi object for the plot")
    voronoi_var: Voronoi = Voronoi(vor_centroids)

    print("Creating the plot")
    fig, ax = plt.subplots(figsize=(14, 10))

    print("Displaying all our data points")
    colormap = plt.get_cmap("tab10")
    colors_true = colormap(var_y.astype(float) / len(np.unique(var_y)))
    ax.scatter(var_x["x_m"], var_x["y_m"],
               c=colors_true, s=30, alpha=0.5, edgecolors='k',
               label="Data points")

    print("Adding Voronoi 2D diagram")
    voronoi_plot_2d(voronoi_var, ax=ax, show_vertices=False,
                    line_colors='red', line_width=1.5,
                    line_alpha=0.7, point_size=2)

    print("Plotting centroids")
    ax.scatter(vor_centroids[:, 0], vor_centroids[:, 1],
               c='black', marker='*', s=150, label='Centroids')

    ax.set_title('Voronoi of DBSCAN Centroids (projected EPSG:6933)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_xlim(var_x["x_m"].min() - 1, var_x["x_m"].max() + 1)
    ax.set_ylim(var_x["y_m"].min() - 1, var_x["y_m"].max() + 1)
    ax.legend()
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


print("1. Let's open the file that contains the dataset")
df: pd.DataFrame = pd.read_csv('smap_soil_moisture_bt.csv')
print(df)
print("")
print("2. Preprocessing")
print("2.1 Turning our coordinates to Euclidean Space and adding to dataframe")
to_ease = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
x_metres: np.ndarray
y_metres: np.ndarray
x_metres, y_metres = to_ease.transform(
    df["longitude"].to_numpy(), df["latitude"].to_numpy())
df["x_m"] = x_metres
df["y_m"] = y_metres
# So all variables have the same weight, we should apply the standard
# scaler to variables in X so all of them are in a similar magnitude
# and therefore similar weight when classifying
print("2.1 Let's apply the StandardScaler to the dataset")
std_sc: StandardScaler = StandardScaler()
df_scaled = std_sc.fit_transform(df)
# approximate neighborhood value change scale
# meters: typical neighborhood radius
print("2.2 Let's add this new columns to our dataframe")
scaled_df = pd.DataFrame(df_scaled,
                         columns=[col + '_scaled' for col in
                                  df.columns])
df = pd.concat([df, scaled_df], axis=1)

df[["x_m_scaled",
    "y_m_scaled",
    "soil_moisture_scaled",
    "brightness_temperature_scaled"]] = df[["x_m_scaled",
                                            "y_m_scaled",
                                            "soil_moisture_scaled",
                                            "brightness_temperature_scaled"]] \
                                        * [1, 1, 1.0, 1.0]

print("6. Estimating best k in eps = eps_m0 * k")
print("6.1 Applying NearestNeighbors algorithm to start estimating eps0")
# k-distance in the *combined* space
nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(
    df[["x_m_scaled",
        "y_m_scaled",
        "soil_moisture_scaled",
        "brightness_temperature_scaled"]])

kdist = np.sort(nn.kneighbors(
    df[["x_m_scaled",
        "y_m_scaled",
        "soil_moisture_scaled",
        "brightness_temperature_scaled"]]
)[0][:, -1])

eps0_m = np.percentile(kdist, 90)
pos: np.ndarray = kdist > 0

print("6.2 We measure our model against a range of values for parameter "
      "k: eps = eps_m0 * k and we will display Silhouette score and "
      "Davies Bouldin index but using a sample of our pixels")
k_range: [float] = [k/10 for k in range(100, 146, 5)]
k_range.sort()
number_clusters_array: [float] = []
number_clusters_array_no_noise: [float] = []
silhouette_scores: [float] = []
davies_bouldin_scores: [float] = []

# We need to sample our pixels as the Silhouette score is
# computationally expensive


for k in k_range:
    print("For K = {0}".format(k))
    dbscan_measure: DBSCAN = DBSCAN(eps=eps0_m * k, min_samples=2,
                                    metric='euclidean')
    dbscan_measure.fit(df[["x_m_scaled",
                           "y_m_scaled",
                           "soil_moisture_scaled",
                           "brightness_temperature_scaled"]])
    print("Number of clusters: ")
    n_clusters: int = len(np.unique(dbscan_measure.labels_))
    number_clusters_array.append(n_clusters)
    if -1 in dbscan_measure.labels_:
        print("That includes a cluster as noise, so it is: {0} ".format(
            n_clusters - 1))
        number_clusters_array_no_noise.append(n_clusters - 1)
    else:
        number_clusters_array_no_noise.append(n_clusters)
    silhouette_avg_: float = silhouette_score(
        df[["x_m_scaled",
            "y_m_scaled",
            "soil_moisture_scaled",
            "brightness_temperature_scaled"]],
        dbscan_measure.labels_)
    print("Avg Silhouette score = {0}".format(silhouette_avg_))
    silhouette_scores.append(silhouette_avg_)
    davies_bouldin_score_: float = davies_bouldin_score(
        df[["x_m_scaled",
            "y_m_scaled",
            "soil_moisture_scaled",
            "brightness_temperature_scaled"]], dbscan_measure.labels_)
    print("Davies Bouldin score = {0}".format(davies_bouldin_score_))
    davies_bouldin_scores.append(davies_bouldin_score_)

plot_inertia_sil_score_db_score(k_range, number_clusters_array,
                                number_clusters_array_no_noise,
                                silhouette_scores,
                                davies_bouldin_scores)

"""
Higher silhouette = better cluster separation and cohesion.
It typically peaks near the “best” number of clusters.
In this case, we see the best values are after k = 13.0. But after a 
while the number of clusters collapse into 1
But these values mean there is some weak data structure

Lower DBI = better clustering (clusters are compact and well-separated).
DBI decreases sharply after k=13.5 however, only one cluster is used

We pick k=12.5 as Silhouette score and Davies-Bouldin index is not that 
far from best while getting 4 clusters (3 removing noise), 13.0 as
metrics improve but we lose 1 cluster (3 clusters, 2 removing noise)
and finally, we pick k=14.5 as it has the best metrics however we only
have 2 clusters (1 cluster plus noise)
"""

print("4. Considering the metrics we picked K={12.5, 13.0, 14.5}")
k_range = [k for k in [12.5, 13.0, 14.5]]
for k in k_range:
    print("5. Building the model for k={0}".format(k))
    dbscan_mod: DBSCAN = DBSCAN(min_samples=2, eps=eps0_m * k,
                                metric="euclidean")
    print("6. Training the model")
    dbscan_mod.fit(df[["x_m_scaled",
                       "y_m_scaled",
                       "soil_moisture_scaled",
                       "brightness_temperature_scaled"]])
    print("7. Getting the labels and cluster_centers")
    labels: np.ndarray = dbscan_mod.labels_

    print("7.1 Computing centroids manually, since DBSCAN "
          "has no cluster_centers_")

    # Extract coordinates or scaled space used for clustering
    x_vals: np.ndarray = df[
        scaled_df.columns].to_numpy()  # same features dbscan trained on

    # Identify valid clusters (excluding noise = -1)
    valid_clusters = [c for c in np.unique(labels) if c != -1]

    centroid_list = []
    for c in valid_clusters:
        pts = x_vals[labels == c]
        centroid_list.append(pts.mean(axis=0))

    centroids = np.vstack(centroid_list) if centroid_list else np.empty(
        (0, x_vals.shape[1]))

    print("7.1 Adding the classification labels to the dataframe")
    df["label"] = labels
    print(df)
    print("")
    print("8. Creating the folium map for k={0}".format(k))
    # Creating random colours for each class
    colors: [str] = ['#{0}'.format(str(hex(rd.randint(0, 4294967296))).
                                   replace("0x", "")) for _ in range(
        len(np.unique(dbscan_mod.labels_)))]

    # Create base Folium map centered at the average location
    m: folium.Map = folium.Map(location=[
        df['latitude'].mean(), df['longitude'].mean()], zoom_start=3)
    print("8.1 Adding each point in the dataset to the world map using "
          "the label to colour them")
    # Add points, color-coded by cluster
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=1,
            color=colors[int(row['label'] % len(colors))],
            fill=True,
            fill_color=colors[int(row['label'] % len(colors))],
            fill_opacity=0.7,
            popup=f'Cluster: {row["label"]}'
        ).add_to(m)

    print("8.2 Adding centroids to the map (computed in lat/lon directly)")

    # Get all non-noise clusters
    valid_clusters = [c for c in np.unique(labels) if c != -1]

    for c in valid_clusters:
        cluster_points = df[df["label"] == c]

        # Mean lat/lon for this cluster
        lat_c = cluster_points["latitude"].mean()
        lon_c = cluster_points["longitude"].mean()

        # Skip if something weird happens
        if np.isnan(lat_c) or np.isnan(lon_c):
            continue

        folium.Marker(
            location=[float(lat_c), float(lon_c)],
            icon=folium.Icon(color='black', icon='star'),
            popup=f'Centroid {c}',
        ).add_to(m)

    # Display map
    m.save("dbscan_world_smap_evaluating_{0}.html".format(k))
    print("")
    silhouette_avg_: float = silhouette_score(
        df[["x_m_scaled",
            "y_m_scaled",
            "soil_moisture_scaled",
            "brightness_temperature_scaled"]],
        labels)
    sample_silhouette_values_: [float] = silhouette_samples(
        df[["x_m_scaled",
            "y_m_scaled",
            "soil_moisture_scaled",
            "brightness_temperature_scaled"]], labels)
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
    plot_silhouette_scores(silhouette_avg_, sample_silhouette_values_, labels,
                           silhouette_score_inter,
                           title_suffix=' DBSCAN Clustering')
    if len(centroids) > 2:
        plot_voronoi_2d(df, labels, centroids, std_sc)
        df.drop(columns=["label"])
