import folium
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# So all variables have the same weight, we should apply the standard
# scaler to variables in X so all of them are in a similar magnitude
# and therefore similar weight when classifying
print("2. Let's apply the StandardScaler to the dataset")
std_sc: StandardScaler = StandardScaler()
scaled_values = std_sc.fit_transform(df)
print("2.1 Let's add this new columns to our dataframe")
# We create a new dataframe where all fields scaled have the suffix
# _scaled
scaled_df = pd.DataFrame(scaled_values,
                         columns=[col + '_scaled' for col in df.columns])
# We add these columns to our current dataframe
df = pd.concat([df, scaled_df], axis=1)

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

print("3.1 We repeat with a higher range")
# Elbow method
k_range: [int] = [k for k in range(2, 20)]
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

print("4. It looks like a range between 4-8 is the right one for k, we"
      " will create maps with K in range (4-8)")
k_range = [k for k in range(4, 9)]
for k in k_range:
    print("5. Building the model for k={0}".format(k))
    kmeans: KMeans = KMeans(init="k-means++", n_clusters=k, n_init=12)
    print("6. Training the model")
    kmeans.fit(df[scaled_df.columns])
    print("7. Getting the labels and cluster_centers")
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("7.1 Adding the classification labels to the dataframe")
    df["label"] = labels
    print(df)
    print("")
    print("8. Creating the folium map for k={0}".format(k))
    # Creating random colours for each class
    colors = ['#{0}'.format(str(hex(rd.randint(0, 4294967296))).
                            replace("0x", "")) for _ in range(
        kmeans.n_clusters)]

    # Create base Folium map centered at the average location
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()],
                   zoom_start=3)
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
    descaled_centroids = std_sc.inverse_transform(centroids)
    # Add centroids with markers
    for i, (lat, lon, sm, bt) in enumerate(descaled_centroids):
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='black', icon='star'),
            popup=f'Centroid {i}'
        ).add_to(m)

    # Display map
    m.save("kmeans_world_smap_k_{0}.html".format(k))
    print("")
