import folium
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, \
    davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from pyproj import Transformer

"""
SOURCE:
All datasets come from the official portal of open data by the 
Urban Planning and Environment Management Department of the 
City Council of Seville

https://cda-idesevilla.opendata.arcgis.com/search
?bbox=-6.022350492431575%2C%2037.33158407745287%2C%20-
5.924503507568425%2C%2037.41234547724453

Bus stops: 
https://cda-idesevilla.opendata.arcgis.com/datasets/ide
SEVILLA::tussam-paradas/explore?location=37.380959%2C-5.930828%2C11.61&
showTable=true


Libraries:
https://cda-idesevilla.opendata.arcgis.com/datasets/
ideSEVILLA::bibliotecas/explore?location=37.375028%2C-5.952682%2C12.70&
showTable=true

Gov Social Centers:
https://cda-idesevilla.opendata.arcgis.com/datasets/
ideSEVILLA::centros-de-servicios-sociales/explore?
location=37.390195%2C-5.958397%2C12.93&showTable=true


Gov Health Clinics:
https://cda-idesevilla.opendata.arcgis.com/datasets/
5fedc02e49ad43ca856cc7a3fda1b809_0/explore?location=
37.383032%2C-5.956702%2C12.30


PROBLEM: We need to classify our bus stops in different groups based
on proximity to public services in this case Libraries, Gov Social 
Centers and Gov Health Clinics using Kmeans algorith
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
                    centroids_: np.ndarray) -> None:
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

    :return: None
    """
    vor_centroids: np.ndarray = np.unique(centroids_, axis=0)

    print("Creating the voronoi object for the plot")
    voronoi_var: Voronoi = Voronoi(vor_centroids[:, [1, 0]])

    print("Creating the plot")
    fig, ax = plt.subplots(figsize=(14, 10))

    print("Displaying all our data points")
    colormap = plt.get_cmap("tab10")
    colors_true = colormap(var_y.astype(float) / len(np.unique(var_y)))
    ax.scatter(var_x["longitude"], var_x["latitude"],
               c=colors_true, s=30, alpha=0.5, edgecolors='k',
               label="Data points")

    print("Adding Voronoi 2D diagram")
    voronoi_plot_2d(voronoi_var, ax=ax, show_vertices=False,
                    line_colors='red', line_width=1.5,
                    line_alpha=0.7, point_size=2)

    print("Plotting centroids")
    ax.scatter(vor_centroids[:, 1], vor_centroids[:, 0],
               c='black', marker='*', s=150, label='Centroids')

    ax.set_title('Voronoi of KMeans Centroids (projected EPSG:3857)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(var_x["longitude"].min() - 0.01, var_x["longitude"].max() +
                0.01)
    ax.set_ylim(var_x["latitude"].min() - 0.01, var_x["latitude"].max() + 0.01)
    ax.legend()
    plt.tight_layout()
    plt.show()


def count_close_services(lat_lon: tuple, df_services: pd.DataFrame,
                         radius_m: float = 500) -> int:
    """
    Counts the number of services in the dataframe df_services at a
    max distance of radius_m from the latitude and longitude passed as
    input

    :param lat_lon: Latitude and longitude of the point to study
    :type lat_lon: tuple
    :param df_services: Dataframe that needs to have a column called
    latitude and another called longitude that we will iterate over
    a check if the distance to the point un lat_lon is less than
    radius_m
    :type df_services: pd.Dataframe
    :param radius_m: Max distance in meters to count the point in
    lat_lon where an entry in df_services will be added to the counter
    :return: Count of services in a radius lower than radius_m
    :rtype: int
    """
    service_count: int = 0
    for _, row_ in df_services.iterrows():
        dist = geodesic(lat_lon, (row_['latitude'], row_['longitude'])).meters
        if dist <= radius_m:
            service_count += 1
    return service_count


print("1. We open our 4 datasets")
# The coordinates in this dataframe are in EPSG:3857 (WebMercator), we
# will rename these columns latitude and longitude and we will convert
# them to this encoding later
bus_stops_sev: pd.DataFrame = pd.read_csv('Sevilla_TUSSAM_Paradas.csv'). \
    rename(columns={"X": "longitude", "Y": "latitude"}).dropna()
libraries_sev: pd.DataFrame = pd.read_csv('Sevilla_Bibliotecas.csv').rename(
    columns={"x": "longitude", "y": "latitude"}
).dropna()
pub_health_clinics_sev: pd.DataFrame = pd.read_csv(
    'Sevilla_Centros_de_Salud.csv'). \
    rename(columns={"x": "longitude", "y": "latitude"}).dropna()
# However in this one the coordinates are latitude and longitude so we
# won't have to convert them
gov_social_serv_sev: pd.DataFrame = pd.read_csv(
    'Sevilla_Centros_de_Servicios_Sociales.csv').rename(
    columns={"Latitud": "latitude", "longitud": "longitude"}).dropna()
print("Bus stops in Seville")
print(bus_stops_sev.columns)
print("")
print("Public libraries in Seville")
print(libraries_sev.columns)
print("")
print("Public Health Clinics in Seville")
print(pub_health_clinics_sev.columns)
print("")
print("Gov Social Services Centers")
print(gov_social_serv_sev.columns)
print("")

# Transformer to latitude and longitude: From EPSG:25830 (UTM zone 30N)
# → EPSG:4326 (lat/lon)
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

print("2. Preprocessing")
print("2.1 Turning all coordinates to longitude and latitude")
# Applying transformer
bus_stops_sev[['longitude', 'latitude']] = bus_stops_sev.apply(
    lambda row_: pd.Series(
        transformer.transform(row_['longitude'], row_['latitude'])),
    axis=1
)
pub_health_clinics_sev[
    ['longitude', 'latitude']] = pub_health_clinics_sev.apply(
    lambda row_: pd.Series(
        transformer.transform(row_['longitude'], row_['latitude'])),
    axis=1
)
libraries_sev[['longitude', 'latitude']] = libraries_sev.apply(
    lambda row_: pd.Series(
        transformer.transform(row_['longitude'], row_['latitude'])),
    axis=1
)

print("2.2 Counting number of libraries in a radius closer than 500 "
      "meters to every bus stop")
bus_stops_sev['n_libraries'] = \
    bus_stops_sev.apply(
        lambda row_: count_close_services((row_['latitude'],
                                           row_['longitude']),
                                          libraries_sev, 500), axis=1)
print("2.3 Counting number of Public Health Centers in a radius closer "
      "than 500 meters to every bus stop")
bus_stops_sev['n_health_centers'] = \
    bus_stops_sev.apply(
        lambda row_: count_close_services((row_['latitude'],
                                           row_['longitude']),
                                          pub_health_clinics_sev, 500), axis=1)
print("2.4 Counting number of Gov Social Services Centers in a radius "
      "closer than 500 meters to every bus stop")
bus_stops_sev['n_social_serv'] = \
    bus_stops_sev.apply(
        lambda row_: count_close_services((row_['latitude'],
                                           row_['longitude']),
                                          gov_social_serv_sev, 500), axis=1)

print("2.5 Getting latitude, longitude and number of libraries, health"
      " centers and social services centers close "
      "for every bus stop to variable X")
df_x: pd.DataFrame = bus_stops_sev[["latitude", "longitude", "n_libraries",
                                    "n_health_centers",
                                    "n_social_serv"]].copy()
print("2.6 Dropping all NA rows")
df_x.dropna(inplace=True)

print("3. We display a plot of inertia for different ranges of k "
      "to apply the Elbow method")
# Elbow method
k_range: [int] = [k for k in range(2, 10)]
inertia: [float] = []
for k in k_range:
    print("For K = {0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    kmeans.fit(df_x)
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
    kmeans.fit(df_x)
    inertia.append(kmeans.inertia_)
    silhouette_avg_: float = silhouette_score(df_x,
                                              kmeans.labels_)
    silhouette_scores.append(silhouette_avg_)
    davies_bouldin_score_: float = davies_bouldin_score(df_x,
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
k ≈ 2–8, after that the improvement is not that big

Higher silhouette = better cluster separation and cohesion.
It typically peaks near the “best” number of clusters.
In this case, we see the best values are around k = 2, 10, 12 or 15
But these values mean there is not a strong structure

Lower DBI = better clustering (clusters are compact and well-separated).
DBI decreases sharply with k=2, k=11-13, k=15-20

As we can see, these 3 measurements give different results, in our 
case I would say are 2, 12 and 15
"""

print("4. It looks like best Ks are 2, 12 and 15, we will create maps "
      "with those values")
k_range = [k for k in [2, 12, 15]]
for k in k_range:
    print("5. Building the model for k={0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    print("6. Training the model")
    kmeans.fit(df_x)
    print("7. Getting the labels and cluster_centers")
    labels: np.ndarray = kmeans.labels_
    centroids: np.ndarray = kmeans.cluster_centers_
    print("7.1 Adding the classification labels to the dataframe")
    df_x["label"] = labels
    print(df_x)
    print("")
    print("8. Creating the folium map for k={0}".format(k))
    # Creating random colours for each class
    colors: [str] = ['#{0}'.format(str(hex(rd.randint(0, 4294967296))).
                                   replace("0x", "")) for _ in range(
        kmeans.n_clusters)]

    # Create base Folium map centered at the average location
    m = folium.Map(location=[df_x['latitude'].mean(),
                             df_x['longitude'].mean()],
                   zoom_start=13)
    print("8.1 Adding each point in the dataset to the world map using "
          "the label to colour them")
    # Add points, color-coded by cluster
    for _, row in df_x.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=6,
            color=colors[int(row['label'] % len(colors))],
            fill=True,
            fill_color=colors[int(row['label'] % len(colors))],
            fill_opacity=0.9,
            popup=f'Cluster: {row["label"]}'
        ).add_to(m)

    print("8.2 Adding centroids to the map")
    centroids_bus: np.ndarray = kmeans.cluster_centers_
    # Add centroids with markers
    for i, (lat, lon, _, _, _) in enumerate(centroids_bus):
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='black', icon='star'),
            popup=f'Centroid {i}'
        ).add_to(m)

    # Display map
    m.save("kmeans_seville_buses_evaluating_{0}.html".format(k))
    print("")
    silhouette_avg_: float = silhouette_score(df_x, labels)
    sample_silhouette_values_: [float] = silhouette_samples(
        df_x, labels)
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
    if centroids.shape[0] > 2:
        plot_voronoi_2d(df_x, labels, centroids)
    else:
        print("Not enough clusters for voronoi")
    df_x.drop(columns=["label"], inplace=True)
