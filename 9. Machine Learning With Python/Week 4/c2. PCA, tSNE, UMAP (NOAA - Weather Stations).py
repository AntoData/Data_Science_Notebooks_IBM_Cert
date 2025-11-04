import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
PROBLEM: We have a several datasets from different weather stations 
across the globe. These files contain climate and weather data including
temperature, elevation...
As classifying by 7 fields might be a little hard and it will be 
impossible to create a visual representation of the different clusters,
we will use Principal Component Analysis (PCA), t-SNE and UMAP to 
reduce the dimensions of each point from 21 to 3 and compare them. 
We will get how much variance these 3 components explain and each 
asteroid translation to these 3 dimensions.
 Then, we will use Kmeans to create clusters and plot a 3D scatter 
 plot of the different asteroids.
SOURCE: 
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2022/
"""

print("1. Opening the file")
# Set the directory containing your downloaded GSOD .csv files
folder_path: str = "./Weather Stations"

# List all .csv files in the directory
csv_files_path: [str] = [
    f for f in os.listdir(folder_path) if f.endswith(".csv")]
print("csv files: {0}".format(csv_files_path))

# Load each file into a DataFrame and collect them in a list
print("Loading each file into the same dataframe")
df_stations: [pd.DataFrame] = []
for file in csv_files_path:
    full_path = os.path.join(folder_path, file)
    try:
        df_station: pd.DataFrame = pd.read_csv(full_path)
        df_station["STATION_FILE"] = file
        df_stations.append(df_station)
    except Exception as e:
        print("Skipped {0} due to error: {1}".format(file, e))

# Combine all data into one DataFrame
var_x: pd.DataFrame = pd.concat(df_stations, ignore_index=True)
print("Loaded {0} files. Final shape: {1}".format(
    len(df_stations), var_x.shape))
print("Content: ")
print(var_x.head())

print("2. Preprocessing, making our variable X dropping the columns "
      "that won't be allowed in PCA")
df_pca_3d_plot: pd.DataFrame = var_x.select_dtypes(include=np.number)
df_pca_3d_plot: pd.DataFrame = df_pca_3d_plot[
    ["ELEVATION", "TEMP", "DEWP", "SLP", "VISIB", "WDSP", "PRCP"]]
x_var: pd.DataFrame = df_pca_3d_plot.fillna(df_pca_3d_plot.mean(
    numeric_only=True))

print("2.1 Applying Standard Scaler")
std_scl: StandardScaler = StandardScaler()
x_var_scaled: pd.DataFrame = std_scl.fit_transform(x_var)

print("3. Creating the PCA model object with 3 components to reduce to "
      "3 dimensions")
pca_model: PCA = PCA(n_components=3)
print("3.1 Training the model")
x_pca: np.ndarray = pca_model.fit_transform(x_var_scaled)
print("3.2 Getting components, variance ratio and the translation of "
      "our variable X to 3 dimensions")
components: np.ndarray = pca_model.components_
variance_ratio_x: np.ndarray = pca_model.explained_variance_ratio_
total_variance_explained: float = variance_ratio_x[:3].sum()
print("Variance ratio explained by component = {0}".format(variance_ratio_x))
print("Total variance ratio explained = {0}".format(total_variance_explained))

print("4. Creating the Kmeans model object")
kmeans_pca_model: KMeans = KMeans(n_clusters=8, random_state=42,
                                  init="k-means++")
print("4.1 Training the model")
kmeans_pca_model.fit(x_pca)
print("4.2 Getting the labels of each asteroid to display them in "
      "different colours according to their cluster")
labels_pca: np.ndarray = kmeans_pca_model.labels_
print("Number of clusters: {0}".format(len(np.unique(labels_pca))))

# Create a DataFrame for Plotly
print("5. Postprocessing, building the dataframe to plot")
df_pca_3d_plot: pd.DataFrame = pd.DataFrame(x_pca, columns=['X', 'Y', 'Z'])

print("6. Creating 3D plot")
# Create interactive 3D scatter plot
fig_pca = px.scatter_3d(df_pca_3d_plot, x='X', y='Y', z='Z',
                        color=labels_pca.astype(str),
                        opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.G10,
                        title="3D Scatter Plot of Asteroids 3 PCAs")

fig_pca.update_traces(marker=dict(size=5, line=dict(width=1, color='black')),
                      showlegend=False)
fig_pca.update_layout(coloraxis_showscale=False, width=1000, height=800)
fig_pca.show()

print("7. We display a plot of inertia for different ranges of k "
      "to apply the Elbow method")
# Elbow method
n_components: [int] = [k for k in range(1, 8)]
variance_ratios_explained: [float] = []
for n_c in n_components:
    pca_model: PCA = PCA(n_components=n_c)
    print("7.1 Training the model for n_components = {0}".format(n_c))
    x_pca: np.ndarray = pca_model.fit_transform(x_var_scaled)
    print("7.2 Getting components, variance ratio and the translation of "
          "our variable X to 3 dimensions")
    components: np.ndarray = pca_model.components_
    variance_ratio_x: np.ndarray = pca_model.explained_variance_ratio_
    total_variance_explained: float = variance_ratio_x[:n_c].sum()
    variance_ratios_explained.append(total_variance_explained)
# Plot
plt.plot(n_components, variance_ratios_explained, 'bo-')
plt.xlabel('Number of components (n)')
plt.ylabel('Total variance explained')
plt.title('Elbow Method')
plt.show()


"""
In this case, the total explained variance is not the best as it is 0.68
In the 3D representation, we can see clear clusters
"""

print("8. Creating the t-SNE model object with 3 components to reduce "
      "to 3 dimensions")
tsne_model: TSNE = TSNE(n_components=3, random_state=42, perplexity=5,
                        max_iter=250)
print("8.1 Training the model")
x_tsne: np.ndarray = tsne_model.fit_transform(x_var_scaled)

print("9. Creating the Kmeans model object")
kmeans_tsne_model: KMeans = KMeans(n_clusters=7, random_state=42,
                                   init="k-means++")
print("9.1 Training the model")
kmeans_tsne_model.fit(x_tsne)
print("9.2 Getting the labels of each asteroid to display them in "
      "different colours according to their cluster")
labels_tsne: np.ndarray = kmeans_tsne_model.labels_
print("Number of clusters: {0}".format(len(np.unique(labels_tsne))))

# Create a DataFrame for Plotly
print("10. Postprocessing, building the dataframe to plot")
df_tsne_3d_plot: pd.DataFrame = pd.DataFrame(x_tsne,
                                             columns=['X', 'Y', 'Z'])

print("11. Creating 3D plot")
# Create interactive 3D scatter plot
fig_tsne = px.scatter_3d(df_tsne_3d_plot, x='X', y='Y', z='Z',
                         color=labels_tsne.astype(str),
                         opacity=0.7,
                         color_discrete_sequence=px.colors.qualitative.G10,
                         title="3D Scatter Plot of Asteroids 3 dim t-SNE")

fig_tsne.update_traces(marker=dict(size=5, line=dict(width=1, color='black')),
                       showlegend=False)
fig_tsne.update_layout(coloraxis_showscale=False, width=1000, height=800)
fig_tsne.show()

"""
In this case, in the 3D representation, we can see the shapes are not
that uniform as in PCA. However, clusters returned by Kmeans are way 
more organised and clear
"""

print("12. Creating the t-SNE model object with 3 components to reduce "
      "to 3 dimensions")
umap_model: UMAP = UMAP(n_components=3, random_state=42,
                        min_dist=0.1,
                        n_jobs=1)

print("12.1 Training the model")
x_umap: np.ndarray = umap_model.fit_transform(x_var_scaled)

print("13. Creating the Kmeans model object")
kmeans_umap_model: KMeans = KMeans(n_clusters=7, random_state=42,
                                   init="k-means++")
print("13.1 Training the model")
kmeans_umap_model.fit(x_umap)
print("13.2 Getting the labels of each asteroid to display them in "
      "different colours according to their cluster")
labels_umap: np.ndarray = kmeans_umap_model.labels_
print("Number of clusters: {0}".format(len(np.unique(labels_umap))))

# Create a DataFrame for Plotly
print("14. Postprocessing, building the dataframe to plot")
df_umap_3d_plot: pd.DataFrame = pd.DataFrame(x_umap,
                                             columns=['X', 'Y', 'Z'])

print("15. Creating 3D plot")
# Create interactive 3D scatter plot
fig_umap = px.scatter_3d(df_umap_3d_plot, x='X', y='Y', z='Z',
                         color=labels_umap.astype(str),
                         opacity=0.7,
                         color_discrete_sequence=px.colors.qualitative.G10,
                         title="3D Scatter Plot of Asteroids 3 dim UMAP")

fig_umap.update_traces(marker=dict(size=5, line=dict(width=1, color='black')),
                       showlegend=False)
fig_umap.update_layout(coloraxis_showscale=False, width=1000, height=800)
fig_umap.show()

"""
In this case, in the 3D representation, we can see the shapes are not
that uniform as in PCA. However, clusters returned by Kmeans are way 
more organised and clear
"""
