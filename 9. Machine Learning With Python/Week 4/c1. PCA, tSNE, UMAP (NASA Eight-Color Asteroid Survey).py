import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

"""
PROBLEM: We have a dataset of contains the reflectance spectra and 
associated data of the Eight Color Asteroid Survey (ECAS), including 
results for 589 asteroids. The file contains 21 columns of different 
photometric (reflectance) data.
As classifying by 21 fields might be a little hard and it will be 
impossible to create a visual representation of the different clusters,
we will use Principal Component Analysis (PCA), t-SNE and UMAP to 
reduce the dimensions of each point from 21 to 3 and compare them. 
We will get how much variance these 3 components explain and each 
asteroid translation to these 3 dimensions.
 Then, we will use DBSCAN to create clusters and plot a 3D scatter 
 plot of the different asteroids.
SOURCE: 
https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.ecas.phot/data/
ecas.tab
"""

print("1. Opening the file")
# Define column names as per the label (21 columns in order):
col_names = [
    "AST_NUMBER", "AST_NAME",
    "S_V", "S_V_STD_DEV", "U_V", "U_V_STD_DEV",
    "B_V", "B_V_STD_DEV", "V_MAG", "V_MAG_STD_DEV",
    "V_W", "V_W_STD_DEV", "V_X", "V_X_STD_DEV",
    "V_P", "V_P_STD_DEV", "V_Z", "V_Z_STD_DEV",
    "OBS_TIME", "CYCLES", "NOTE"
]

# Define the fixed column widths (including the spacing between fields):
col_widths = [7, 18, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 20, 4, 2]

# Read the fixed-width file:
df_pca_3d_plot = pd.read_fwf("ecas.tab", names=col_names, widths=col_widths,
                             na_values=["-9.999", "-99", "-9", "---"])

print("2. Preprocessing, making our variable X dropping the columns "
      "that won't be allowed in PCA")
x_var: pd.DataFrame = df_pca_3d_plot.drop(columns=["AST_NUMBER", "AST_NAME",
                                                   "OBS_TIME"])
x_var = x_var.fillna(x_var.mean(numeric_only=True))

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

print("4. Creating NearestNeighbors to get a valid eps for DBSCAN")
nn_pca: NearestNeighbors = NearestNeighbors(n_neighbors=2,
                                            metric="euclidean").fit(x_pca)
kdist_pca: np.ndarray = np.sort(nn_pca.kneighbors(x_pca)[0][:, -1])
eps0_m_pca: float = np.percentile(kdist_pca, 90)

print("5. Creating the DBSCAN model object")
dbscan_pca_model: DBSCAN = DBSCAN(eps=eps0_m_pca, metric="euclidean",
                                  min_samples=2)
print("5.1 Training the model")
dbscan_pca_model.fit(x_pca)
print("5.2 Getting the labels of each asteroid to display them in "
      "different colours according to their cluster")
labels_pca: np.ndarray = dbscan_pca_model.labels_
print("Number of clusters: {0}".format(len(np.unique(labels_pca))))

# Create a DataFrame for Plotly
print("6. Postprocessing, building the dataframe to plot")
df_pca_3d_plot: pd.DataFrame = pd.DataFrame(x_pca, columns=['X', 'Y', 'Z'])

print("7. Creating 3D plot")
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


print("7.1 Displaying the variance ratio explain by x components")
print("3. We display a plot of inertia for different ranges of k "
      "to apply the Elbow method")
# Elbow method
n_components: [int] = [k for k in range(1, 9)]
variance_ratios_explained: [float] = []
for n_c in n_components:
    pca_model: PCA = PCA(n_components=n_c)
    print("3.1 Training the model for n_components = {0}".format(n_c))
    x_pca: np.ndarray = pca_model.fit_transform(x_var_scaled)
    print("3.2 Getting components, variance ratio and the translation of "
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
In this case, the total explained variance is not the best as it is 0.66
In the 3D representation, we can see most asteroids fall within the same
cluster/blob but the other clusters are more dispersed
"""

print("8. Creating the t-SNE model object with 3 components to reduce "
      "to 3 dimensions")
tsne_model: TSNE = TSNE(n_components=3, random_state=42, perplexity=5,
                        max_iter=250)
print("8.1 Training the model")
x_tsne: np.ndarray = tsne_model.fit_transform(x_var_scaled)

print("9. Creating NearestNeighbors to get a valid eps for DBSCAN")
nn_tsne: NearestNeighbors = NearestNeighbors(n_neighbors=2,
                                             metric="euclidean").fit(x_tsne)
kdist_tsne: np.ndarray = np.sort(nn_tsne.kneighbors(x_tsne)[0][:, -1])
eps0_m_tsne: float = np.percentile(kdist_tsne, 90)

print("10. Creating the DBSCAN model object")
dbscan_tsne_model: DBSCAN = DBSCAN(eps=eps0_m_tsne, metric="euclidean",
                                   min_samples=2)
print("10.1 Training the model")
dbscan_tsne_model.fit(x_tsne)
print("10.2 Getting the labels of each asteroid to display them in "
      "different colours according to their cluster")
labels_tsne: np.ndarray = dbscan_tsne_model.labels_
print("Number of clusters: {0}".format(len(np.unique(labels_tsne))))

# Create a DataFrame for Plotly
print("11. Postprocessing, building the dataframe to plot")
df_tsne_3d_plot: pd.DataFrame = pd.DataFrame(x_tsne,
                                             columns=['X', 'Y', 'Z'])

print("12. Creating 3D plot")
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
In this case, in the 3D representation, we can see clusters don't create
 close blobs
"""

print("13. Creating the t-SNE model object with 3 components to reduce "
      "to 3 dimensions")
umap_model: UMAP = UMAP(n_components=3, random_state=42,
                        min_dist=0.1,
                        n_jobs=1)

print("13.1 Training the model")
x_umap: np.ndarray = umap_model.fit_transform(x_var_scaled)

print("14. Creating NearestNeighbors to get a valid eps for DBSCAN")
nn_umap: NearestNeighbors = NearestNeighbors(n_neighbors=2,
                                             metric="euclidean").fit(x_umap)
kdist_umap: np.ndarray = np.sort(nn_umap.kneighbors(x_umap)[0][:, -1])
eps0_m_umap: float = np.percentile(kdist_umap, 90)

print("15. Creating the DBSCAN model object")
dbscan_umap_model: DBSCAN = DBSCAN(eps=eps0_m_umap, metric="euclidean",
                                   min_samples=2)
print("15.1 Training the model")
dbscan_umap_model.fit(x_umap)
print("15.2 Getting the labels of each asteroid to display them in "
      "different colours according to their cluster")
labels_umap: np.ndarray = dbscan_umap_model.labels_
print("Number of clusters: {0}".format(len(np.unique(labels_umap))))

# Create a DataFrame for Plotly
print("16. Postprocessing, building the dataframe to plot")
df_umap_3d_plot: pd.DataFrame = pd.DataFrame(x_umap,
                                             columns=['X', 'Y', 'Z'])

print("17. Creating 3D plot")
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
In this case, in the 3D representation, we can see clusters don't create
 close blobs either
"""
