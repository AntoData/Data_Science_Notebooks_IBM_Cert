import os
import rasterio
import numpy as np
import pandas as pd
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

"""
We have multiple instances of different classes of low resolution multi 
spectrum images in tiff file. We will use PCA to reduce their dimensions
until we get a total explained variance of 0.80
Then, we will use that dataset to reduce to 2 dimensions using t-SNE 
and also UMAP and use a Kmeans algorithm to see how it goes
We will display the datasets after t-SNE and UMAP in two plots with two 
figures, one figure will display the points with their original labels 
and the other with the labels predicted by Kmeans. We will also get 
accuracy scores for both t-SNE and UMAP
Then, we will use PCA over the original dataset to reduce to 3 
dimensions. We will use a Kmeans model to predict which the labels 
should be and display two figures in a 3D scatter. We will get the 
accuracy score too
Finally, we will use another Kmeans model with the original dataset 
with its original dimensions to predict the labels and get accuracy 
score

SOURCE: https://zenodo.org/records/7711810#:~:text=EuroSAT_MS
"""

print("1. Let's open the files")

# Folder containing your TIFF files
tiff_folder: str = "./EuroSat files"

# Land-cover class labels based on filename
class_labels: {str: int} = {
    'AnnualCrop': 0,
    'Forest': 1,
    'Highway': 2,
    'Pasture': 3,
    'PermanentCrop': 4,
    'Residential': 5,
    'River': 6,
    'SeaLake': 7
}
print("Categories/Labels = {0}".format(class_labels))
print("Files in folder to open")
print(os.listdir(tiff_folder))
# List all TIFF files in the folder
tiff_files: [str] = [f for f in os.listdir(tiff_folder) if f.endswith('.tif')]

# Initialize an empty list to store dataframes for each TIFF file
dfs: [pd.DataFrame] = []

# Iterate over each TIFF file
for tiff_file in tiff_files:
    # Build the full file path
    tiff_path: str = os.path.join(tiff_folder, tiff_file)
    print("1.1 Opening file = {0}".format(tiff_path))
    # Extract the land-cover class from the filename
    class_name: str = tiff_file.split('_')[
        0]  # e.g., "AnnualCrop" from "AnnualCrop_1.tiff"
    print("1.2 Getting class name = {0}".format(class_name))
    class_label: str = class_labels.get(
        class_name, -1)  # Use the label, default to -1 if not found

    print("1.3 Opening file with rasterio")
    # Open the TIFF file using rasterio
    with rasterio.open(tiff_path) as src:
        # Read the image data as a numpy array (shape: bands, height,
        # width)
        print("1.4 Reading file and turning it into a np.ndarray")
        image_array: np.ndarray = src.read()

        # Flatten the array into 2D: (height * width, bands)
        print("1.5 Converting the array into 2D")
        flattened_array: np.ndarray = \
            image_array.reshape((-1, image_array.shape[0])).T

        print("1.6 Turning the array into a DataFrame")
        # Convert the flattened array to a pandas DataFrame
        df: pd.DataFrame = pd.DataFrame(flattened_array)

        # Add the class label to the dataframe (create a new column
        # for the label)
        print("1.7 Adding class label column")
        df['class_label'] = class_label

        print("1.8 Adding the file name to the dataframe")
        df['filename'] = tiff_file

        # Append the dataframe to the list of dataframes
        print("1.9 Appending dataframe to list")
        dfs.append(df)

# Concatenate all dataframes into one large dataframe
# (if you want them combined)
print("1.10 Merging all dataframes into a final DataFrame")
final_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

# Show the first few rows of the dataframe
print(final_df.head())
print("2. Preprocessing")
print("2.1 Getting variable x")
var_x: pd.DataFrame = final_df.drop(columns=["class_label", "filename"])
print("2.2 Getting variable y (labels)")
var_y: pd.DataFrame = final_df[["class_label"]]

print("2.3 Using standard scaler in variable x")
var_x_scaled: np.ndarray = StandardScaler().fit_transform(var_x)

print("3. Creating our PCA model so total explained variance is around 0.80")
pca_model: PCA = PCA(n_components=0.80)
print("3.1 Training the model and getting our variable x now with "
      "reduced dimensions")
x_pca: np.ndarray = pca_model.fit_transform(var_x_scaled)
print(x_pca)
print("3.2 Total explained variance = {0}".format(sum(
    pca_model.explained_variance_ratio_)))

print("4. Plotting cumulative explained variance")
# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.show()
# Convert PCA output to a DataFrame

print("5. Converting our variable x after PCA into a Dataframe")
x_pca_df: pd.DataFrame = pd.DataFrame(x_pca)

print("5.1 Splitting our variable into training and testing sets")
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, var_y["class_label"], test_size=0.2, random_state=42)

print("6. Splitting our variable into training and testing sets "
      "(grouped by image)")
groups: pd.Series = final_df["filename"]

split: GroupShuffleSplit = GroupShuffleSplit(test_size=0.2, n_splits=1,
                                             random_state=42)
train_idx, test_idx = next(split.split(x_pca, var_y["class_label"],
                                       groups=groups))

x_train_clf: np.ndarray = x_pca[train_idx]
x_test_clf: np.ndarray = x_pca[test_idx]
y_train_clf: pd.Series = var_y["class_label"].iloc[train_idx]
y_test_clf: pd.Series = var_y["class_label"].iloc[test_idx]

print("6.1 Creating a RandomForestClassifier "
      "(group-aware split to avoid leakage)")
clf: RandomForestClassifier = RandomForestClassifier(n_estimators=100,
                                                     random_state=42)

print("6.2 Training the model with x and y train")
clf.fit(x_train_clf, y_train_clf)

print("6.3 Getting prediction for x test")
y_pred_clf: np.ndarray = clf.predict(x_test_clf)

print("6.4 Getting the accuracy score")
accuracy_clf: float = accuracy_score(y_test_clf, y_pred_clf)
print("Classification accuracy (grouped by filename): {0}".format(
    accuracy_clf))

print("7. Creating the t-SNE object with 2 components")
tsne_model: TSNE = TSNE(n_components=2, random_state=42)
print("7.1 Training the model with our variable x after PCA")
x_tsne: np.ndarray = tsne_model.fit_transform(x_pca_df)

print("8. Creating the Kmeans model object")
kmeans_tsne_model: KMeans = KMeans(n_clusters=len(
    var_y["class_label"].unique()), random_state=42,
    init="k-means++")
print("8.1 Training the model")
kmeans_tsne_model.fit(x_tsne)
print("8.2 Getting the labels to display them in "
      "different colours according to their cluster")
labels_tsne: np.ndarray = kmeans_tsne_model.labels_
print("8.3 Number of clusters: {0}".format(len(np.unique(labels_tsne))))
print("8.4 Getting accuracy score")
accuracy: float = accuracy_score(var_y["class_label"], labels_tsne)
print("Accuracy score = {0}".format(accuracy))

print("9. Plotting original dataset in 2D after PCA and t-SNE "
      "with their original labels vs labels predicted by Kmeans")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc1 = ax1.scatter(x_tsne[:, 0], x_tsne[:, 1],
                  c=var_y["class_label"],
                  cmap='tab10', s=10)
fig.colorbar(sc1, ax=ax1, label='Class Label')
ax1.set_title('t-SNE Embedding of EuroSAT Data')

sc2 = ax2.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels_tsne, cmap='tab10',
                  s=10)
fig.colorbar(sc2, ax=ax2, label='Class Label')
ax2.set_title('t-SNE Kmeans result of EuroSAT Data')
plt.show()
print("")

print("10. Creating UMAP object with 2 components")
umap_model: UMAP = UMAP(n_components=2, random_state=42)
print("10.1 Training model with x after PCA")
x_umap: np.ndarray = umap_model.fit_transform(x_pca_df)

print("11. Creating the Kmeans model object")
kmeans_umap_model: KMeans = KMeans(n_clusters=len(
    var_y["class_label"].unique()), random_state=42,
    init="k-means++")
print("11.1 Training the model")
kmeans_umap_model.fit(x_umap)
print("11.2 Getting the labels to display them in "
      "different colours according to their cluster")
labels_umap: np.ndarray = kmeans_umap_model.labels_
print("11.3 Number of clusters: {0}".format(len(np.unique(labels_umap))))
print("11.4 Getting accuracy score")
accuracy: float = accuracy_score(var_y["class_label"], labels_umap)
print('Accuracy score: {0}'.format(accuracy))

print("12. Plotting original dataset in 2D after PCA and UMAP "
      "with their original labels vs labels predicted by Kmeans")
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

sc3 = ax3.scatter(x_umap[:, 0], x_umap[:, 1], c=var_y["class_label"],
                  cmap='tab10', s=10)
fig2.colorbar(sc3, ax=ax3, label='Class Label')
ax3.set_title('UMAP Embedding of EuroSAT Data')

sc4 = ax4.scatter(x_umap[:, 0], x_umap[:, 1], c=labels_umap, cmap='tab10',
                  s=10)
fig2.colorbar(sc4, ax=ax4, label='Class Label')
ax4.set_title('UMAP Kmeans result of EuroSAT Data')
plt.show()

print("13. Creating the PCA model object with 3 components to reduce to "
      "3 dimensions")
pca_model: PCA = PCA(n_components=3)
print("13.1 Training the model")
x_pca: np.ndarray = pca_model.fit_transform(var_x_scaled)
print("13.2 Getting components, variance ratio and the translation of "
      "our variable X to 3 dimensions")
components: np.ndarray = pca_model.components_
variance_ratio_x: np.ndarray = pca_model.explained_variance_ratio_
total_variance_explained: float = variance_ratio_x[:3].sum()
print("13.3 Variance ratio explained by component = {0}".format(
    variance_ratio_x))
print("13.4 Total variance ratio explained = {0}".format(
    total_variance_explained))

print("14. Postprocessing, building the dataframe to plot")
df_pca_3d_plot: pd.DataFrame = pd.DataFrame(x_pca, columns=['X', 'Y', 'Z'])
df_pca_3d_plot['label'] = np.asarray(var_y).ravel().astype(
    str)  # attach labels here

# Quick sanity check
assert len(df_pca_3d_plot) == len(df_pca_3d_plot['label'])

print("15. Creating the Kmeans object with n_clusters = {0} and getting "
      "predicted labels".format(len(var_y["class_label"].unique())))
kmeans_labels: np.ndarray = KMeans(n_clusters=len(
    var_y["class_label"].unique()), random_state=42,
    init="k-means++").fit_predict(x_pca)

print("16. Plotting original dataset in 3D after PCA to 3 dimensions  "
      "with their original labels vs labels predicted by Kmeans")
# --- Create subplots layout: two 3D scenes ---
fig3 = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("PCA - True Labels", "PCA - KMeans Clusters")
)

# --- Left: PCA colored by true labels ---
fig3.add_trace(
    go.Scatter3d(
        x=x_pca[:, 0],
        y=x_pca[:, 1],
        z=x_pca[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=var_y["class_label"],
            colorscale='Viridis',
            opacity=0.8
        ),
        name="True Labels"
    ),
    row=1, col=1
)

# --- Right: PCA colored by KMeans clusters ---
fig3.add_trace(
    go.Scatter3d(
        x=x_pca[:, 0],
        y=x_pca[:, 1],
        z=x_pca[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=kmeans_labels,
            colorscale='Viridis',
            opacity=0.8
        ),
        name="KMeans Clusters"
    ),
    row=1, col=2
)
fig3.show()

print("17. Creating Kmeans object")
kmeans_all_dimension: KMeans = KMeans(n_clusters=len(
    var_y["class_label"].unique()), init="k-means++", random_state=42)
print("17.1 Training the model with original variable x scaled")
kmeans_all_dimension.fit_transform(var_x_scaled)
print("17.2 Getting predictions of labels by the model")
y_pred: np.ndarray = kmeans_all_dimension.labels_
accuracy = accuracy_score(var_y["class_label"], y_pred)
print("17.3 Accuracy of Kmeans with the original dataset with all "
      "dimensions = {0}".format(accuracy))
