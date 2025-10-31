import random as rd
import folium
import pandas as pd
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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
print("1. Let's open the file that contains the dataset")
df: pd.DataFrame = pd.read_csv('smap_soil_moisture_bt.csv')
print(df)
print("")
print("2. We need to turn the latitude and longitude to meters")
# Converter object from degrees to meters
to_ease = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
# Converting degrees to meters
x_m, y_m = to_ease.transform(df["longitude"].to_numpy(),
                             df["latitude"].to_numpy())
df["x_m"] = x_m
df["y_m"] = y_m

# So all variables have the same weight, we should apply the standard
# scaler to variables in X so all of them are in a similar magnitude
# and therefore similar weight when classifying
print("2.1 Let's apply the StandardScaler to the dataset")
df_scaled = StandardScaler().fit_transform(df)
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
                                        * [0.75, 0.75, 2.0, 2.0]

nn = NearestNeighbors(n_neighbors=2,
                      metric="euclidean").fit(
    df[["x_m_scaled",
        "y_m_scaled",
        "soil_moisture_scaled",
        "brightness_temperature_scaled"]])
kdist = np.sort(nn.kneighbors(df[["x_m_scaled",
                                  "y_m_scaled",
                                  "soil_moisture_scaled",
                                  "brightness_temperature_scaled"]])[0][:, -1])

eps0_m = np.percentile(kdist, 90)
print("Eps0_m = {0}".format(eps0_m))
print("3. We will apply a multiplier k = [100, 105, 110]")
k_range = [k/100 for k in range(100, 115, 5)]
for k in k_range:
    print("k = {0}".format(k))
    print("eps0_m * k = {0}".format(eps0_m * k))
    print("4. Creating the HDBSCAN model")
    hdbscan_mod: hdbscan.HDBSCAN = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",   # merges → fewer gaps
        cluster_selection_epsilon=float(eps0_m * k),
        prediction_data=True
    )
    print("4.1 Training such model")
    hdbscan_mod.fit(df[["x_m_scaled", "y_m_scaled", "soil_moisture_scaled",
                       "brightness_temperature_scaled"]])
    print("4.2 Getting the labels and cluster_centers")
    labels = hdbscan_mod.labels_
    print("4.2 Adding the classification labels to the dataframe")
    df["label"] = labels
    print(df)
    n_labels: int = len(df['label'].unique())
    print("Number of clusters = {0}".format(n_labels))
    print("")
    print("5. Creating the folium map for k={0}".format(n_labels))
    # Creating random colours for each class
    colors = ["#{:06x}".format(rd.randint(0, 0xFFFFFF)) for _ in range(
        len(df["label"].unique()))]

    labels: [] = df["label"].unique()
    labels.sort()
    color_map: dict = {label: colors[i] for i, label in enumerate(labels)}

    # Create base Folium map centered at the average location
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()],
                   zoom_start=3)
    print("5.1 Adding each point in the dataset to the world map using "
          "the label to colour them")
    # Add points, color-coded by cluster
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=1,
            color=color_map[row['label']],
            fill=True,
            fill_color=color_map[row['label']],
            fill_opacity=0.7,
            popup=f'Cluster: {row["label"]}'
        ).add_to(m)

    # Display map
    m.save("hdbscan_world_smap_k_{0}.html".format(n_labels))
    print("")
