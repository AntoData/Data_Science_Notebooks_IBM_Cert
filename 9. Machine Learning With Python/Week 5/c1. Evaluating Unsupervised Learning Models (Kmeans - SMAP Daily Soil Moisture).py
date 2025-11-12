import folium
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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


def plot_inertia_sil_score_db_score(k_values: [int], inertias_: [float],
                                    silhouette_scores_: [float],
                                    davies_bouldin_indexes_: [float]) -> None:
    """
    Plots in 3 different graphs the inertia, avg silhouette scores and
    Davies Boulding indexes across K (number of clusters in Kmeans)

    :param k_values: Array with the different number of clusters that
    were used in the experiment
    :type k_values: [int]
    :param inertias_: Array with the inertia values we got for the K
    values in the previous variable
    :type inertias_: [float]
    :param silhouette_scores_: Avg Silhouette scores for the K values in
    k_values in a Kmeans model
    :type silhouette_scores_: [float]
    :param davies_bouldin_indexes_: Davies Boulding indexes for the K
    values in k_values in a Kmeans model
    :type davies_bouldin_indexes_: [float]
    :return: None
    """

    # Plot the inertia values (Elbow Method)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(k_values, inertias_, marker='o')
    plt.title('Elbow Method: Inertia vs. k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    # Plot silhouette scores
    plt.subplot(1, 3, 2)
    plt.plot(k_values, silhouette_scores_, marker='o')
    plt.title('Silhouette Score vs. k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    # Plot Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(k_values, davies_bouldin_indexes_, marker='o')
    plt.title('Davies-Bouldin Index vs. k')
    plt.xlabel('Number of Clusters (k)')
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

    all_scaled_cols = [c for c in df.columns if c.endswith('_scaled')]
    x_index: int = all_scaled_cols.index("x_scaled")
    y_index: int = all_scaled_cols.index("y_scaled")

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
    ax.scatter(var_x["x"], var_x["y"],
               c=colors_true, s=30, alpha=0.5, edgecolors='k',
               label="Data points")

    print("Adding Voronoi 2D diagram")
    voronoi_plot_2d(voronoi_var, ax=ax, show_vertices=False,
                    line_colors='red', line_width=1.5,
                    line_alpha=0.7, point_size=2)

    print("Plotting centroids")
    ax.scatter(vor_centroids[:, 0], vor_centroids[:, 1],
               c='black', marker='*', s=150, label='Centroids')

    ax.set_title('Voronoi of KMeans Centroids (projected EPSG:3857)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_xlim(var_x["x"].min() - 1, var_x["x"].max() + 1)
    ax.set_ylim(var_x["y"].min() - 1, var_x["y"].max() + 1)
    ax.legend()
    plt.tight_layout()
    plt.show()


print("1. Let's open the file that contains the dataset")
df: pd.DataFrame = pd.read_csv('smap_soil_moisture_bt.csv')
print(df)
print("")
print("2. Preprocessing")
print("2.1 Turning our coordinates to Euclidean Space and adding to dataframe")
from_coor_to_m: Transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857",
                                                   always_xy=True)
x_metres: np.ndarray
y_metres: np.ndarray
x_metres, y_metres = from_coor_to_m.transform(
    df["longitude"].to_numpy(), df["latitude"].to_numpy())
df["x"] = x_metres
df["y"] = y_metres
# So all variables have the same weight, we should apply the standard
# scaler to variables in X so all of them are in a similar magnitude
# and therefore similar weight when classifying
print("2.2 Let's apply the StandardScaler to the dataset")
std_sc: StandardScaler = StandardScaler()
scaled_values: np.ndarray = std_sc.fit_transform(df.drop(
    columns=["latitude", "longitude"]))
print("2.3 Let's add this new columns to our dataframe")
# We create a new dataframe where all fields scaled have the suffix
# _scaled
feature_cols: [] = df.drop(columns=["latitude", "longitude"]).columns
scaled_df: pd.DataFrame = pd.DataFrame(scaled_values,
                                       columns=[col + '_scaled' for col in
                                                feature_cols])
# We add these columns to our current dataframe
df: pd.DataFrame = pd.concat([df, scaled_df], axis=1)

print("3. We display a plot of inertia for different ranges of k "
      "to apply the Elbow method")
# Elbow method
k_range: [int] = [k for k in range(2, 10)]
inertia: [float] = []
for k in k_range:
    print("For K = {0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    kmeans.fit(df[scaled_df.columns])
    inertia.append(kmeans.inertia_)

# Plot
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method')
plt.show()

print("3.1 We repeat with a higher range and add Silhouette score and "
      "Davies Boulding index")
# Elbow method
k_range: [int] = [k for k in range(2, 21)]
inertia: [float] = []
silhouette_scores: [float] = []
davies_bouldin_scores: [float] = []
for k in k_range:
    print("For K = {0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    kmeans.fit(df[scaled_df.columns])
    inertia.append(kmeans.inertia_)
    silhouette_avg_: float = silhouette_score(df[scaled_df.columns],
                                              kmeans.labels_)
    silhouette_scores.append(silhouette_avg_)
    davies_bouldin_score_: float = davies_bouldin_score(df[scaled_df.columns],
                                                        kmeans.labels_)
    davies_bouldin_scores.append(davies_bouldin_score_)

plot_inertia_sil_score_db_score(k_range, inertia, silhouette_scores,
                                davies_bouldin_scores)

"""
Inertia measures the total within-cluster sum of squared distances 
— lower is better.
However, inertia always decreases as you add clusters, so you look for 
the “elbow point” — where adding more clusters doesn't give a big 
improvement.
In this case, we can see the biggest improvement happens between 
k ≈ 5–6, after that the improvement is not that big

Higher silhouette = better cluster separation and cohesion.
It typically peaks near the “best” number of clusters.
In this case, we see the best values are around k = 7–8 (≈ 0.39)
But these values mean there is not a strong structure

Lower DBI = better clustering (clusters are compact and well-separated).
DBI decreases sharply from k=2 to around k=8–10, then levels off 
around 0.9–1.0 after k≈10. Best values are between 10-12

As we can see, these 3 measurements give different results, in our 
case I would say are 7, 10 and 17
"""

print("4. It looks like best Ks are 7, 10 and 17 is the right one for k,"
      " we will create maps with those values")
k_range = [k for k in [7, 10, 17]]
for k in k_range:
    print("5. Building the model for k={0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    print("6. Training the model")
    kmeans.fit(df[scaled_df.columns])
    print("7. Getting the labels and cluster_centers")
    labels: np.ndarray = kmeans.labels_
    centroids: np.ndarray = kmeans.cluster_centers_
    print("7.1 Adding the classification labels to the dataframe")
    df["label"] = labels
    print(df)
    print("")
    print("8. Creating the folium map for k={0}".format(k))
    # Creating random colours for each class
    colors: [str] = ['#{0}'.format(str(hex(rd.randint(0, 4294967296))).
                            replace("0x", "")) for _ in range(
        kmeans.n_clusters)]

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

    print("8.2 Adding centroids to the map")
    print("8.2.1 Transforming centroids from scaled to original scale")
    tfm_back: Transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326",
                                                 always_xy=True)

    descaled_cen: np.ndarray = std_sc.inverse_transform(
        centroids)
    print("8.2.2 Converting centroids from x and y in metres to "
          "latitude and longitude in degrees")
    x_idx: int = scaled_df.columns.get_loc("x_scaled")
    y_idx: int = scaled_df.columns.get_loc("y_scaled")
    x_centroids: np.ndarray = descaled_cen[:, x_idx]
    y_centroids: np.ndarray = descaled_cen[:, y_idx]

    lon_cen: np.ndarray
    lat_cen: np.ndarray
    lon_cen, lat_cen = tfm_back.transform(x_centroids, y_centroids)
    centroids_coord: np.ndarray = np.column_stack((lat_cen, lon_cen))
    # Add centroids with markers
    for i, (lat, lon) in enumerate(centroids_coord):
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='black', icon='star'),
            popup=f'Centroid {i}'
        ).add_to(m)

    # Display map
    m.save("kmeans_world_smap_evaluating_{0}.html".format(k))
    print("")
    silhouette_avg_: float = silhouette_score(df[scaled_df.columns], labels)
    sample_silhouette_values_: [float] = silhouette_samples(
        df[scaled_df.columns], labels)
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
                           title_suffix=' k-Means Clustering')

    plot_voronoi_2d(df, labels, centroids, std_sc)
