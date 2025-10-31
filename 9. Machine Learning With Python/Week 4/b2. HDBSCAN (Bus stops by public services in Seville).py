import folium
import hdbscan
import numpy as np
import pandas as pd
import random as rd
from geopy.distance import geodesic
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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
Centers and Gov Health Clinics using HDBSCAN algorith
"""


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

print("2. Turning all coordinates to longitude and latitude")
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

print("3.1 Counting number of libraries in a radius closer than 500 "
      "meters to every bus stop")
bus_stops_sev['n_libraries'] = \
    bus_stops_sev.apply(
        lambda row_: count_close_services((row_['latitude'],
                                           row_['longitude']),
                                          libraries_sev, 500), axis=1)
print("3.2 Counting number of Public Health Centers in a radius closer "
      "than 500 meters to every bus stop")
bus_stops_sev['n_health_centers'] = \
    bus_stops_sev.apply(
        lambda row_: count_close_services((row_['latitude'],
                                           row_['longitude']),
                                          pub_health_clinics_sev, 500), axis=1)
print("3.3 Counting number of Gov Social Services Centers in a radius "
      "closer than 500 meters to every bus stop")
bus_stops_sev['n_social_serv'] = \
    bus_stops_sev.apply(
        lambda row_: count_close_services((row_['latitude'],
                                           row_['longitude']),
                                          gov_social_serv_sev, 500), axis=1)

print("4. Preprocessing the data: Getting latitude, longitude and number"
      " of libraries, health centers and social services centers close "
      "for every bus stop to variable X")
df_x: pd.DataFrame = bus_stops_sev[["latitude", "longitude", "n_libraries",
                                    "n_health_centers",
                                    "n_social_serv"]].copy()

print("4.1 Dropping all NA rows")
df_x.dropna(inplace=True)

print("5.1 Let's apply the StandardScaler to the dataset")
df_scaled = StandardScaler().fit_transform(df_x)
# approximate neighborhood value change scale
# meters: typical neighborhood radius
print("5.2 Let's add this new columns to our dataframe")
scaled_df = pd.DataFrame(df_scaled,
                         columns=[col + '_scaled' for col in
                                  df_x.columns])
df_x = pd.concat([df_x, scaled_df], axis=1)

print("6. Building NearestNeighbors to get distances between")
# k-distance in the *combined* space
nn = NearestNeighbors(n_neighbors=2,
                      metric="euclidean").fit(
    df_x[["latitude_scaled", "longitude_scaled", "n_libraries_scaled",
          "n_health_centers_scaled",
          "n_social_serv_scaled"]])
print("6.1 We get the distances between points sorted")
kdist = np.sort(
    nn.kneighbors(df_x[["latitude_scaled", "longitude_scaled",
                        "n_libraries_scaled", "n_health_centers_scaled",
                        "n_social_serv_scaled"]])[0][:, -1])

print("6.2 Getting percentile 90 of distances between points")
eps0_m = np.percentile(kdist, 90)

print("7. Iterating through different multipliers of the initial epsilon")
k_range: [int] = [k for k in range(95, 99)]
for k in k_range:
    print("7.1 Multiplier k = {0}".format(k))
    eps_m = float(eps0_m * k)
    print("7.2 Building the HDBSCAN object")
    hdbscan_mod: hdbscan.HDBSCAN = hdbscan.HDBSCAN(
        min_samples=2,
        metric="euclidean",
        min_cluster_size=2,
        cluster_selection_method="eom",
        # merges → fewer gaps
        cluster_selection_epsilon=float(
            eps0_m * k),
        prediction_data=True
    )
    print("7.3 Training the model")
    hdbscan_mod.fit(df_x[["latitude_scaled", "longitude_scaled",
                          "n_libraries_scaled", "n_health_centers_scaled",
                          "n_social_serv_scaled"]])
    print("7.4 Getting labels for each point")
    df_x["label"] = hdbscan_mod.labels_
    print("8. Creating map")
    m = folium.Map(location=[df_x['latitude'].mean(),
                             df_x['longitude'].mean()],
                   zoom_start=13)
    print("8.1 Creating colours dynamically")
    colors = ["#{:06x}".format(rd.randint(0, 0xFFFFFF)) for _ in range(
        len(df_x["label"].unique()))]
    labels: [] = df_x["label"].unique()
    labels.sort()
    color_map: dict = {label: colors[i] for i, label in enumerate(labels)}

    print("9.1 Adding each point in the dataset to the world map using "
          "the label to colour them")
    # Add points, color-coded by cluster
    for _, row in df_x.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=6,
            color=color_map[row['label']],
            fill=True,
            fill_color=color_map[row['label']],
            fill_opacity=1.0,
            popup=f'Cluster: {row["label"]}'
        ).add_to(m)

    # Display map
    m.save("hdbscan_seville_bus_stops_grous_{0}.html".format(k))
    print("")
