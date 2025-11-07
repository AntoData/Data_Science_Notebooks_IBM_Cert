import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, \
    precision_score, recall_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

"""
Source: https://data.nasa.gov/dataset/meteorite-landings

Independent variable x
Columns:
GeoLocation: Divided in
Latitude: Geographical coordinates of the meteorite landing. These can 
help identify regional patterns in meteorite distribution.
Longitude: Same as above

Year: The year the meteorite was found or observed falling. This can 
capture temporal trends in meteorite discoveries.

Fall Status: Indicates whether the meteorite was observed falling 
("Fell") or was found later ("Found"). This categorical variable can 
be encoded numerically for modeling.

Recclass: The classification of the meteorite, indicating its 
composition. This categorical variable can be transformed using 
one-hot encoding to be used in the model.
"""


def align_kmeans_labels(y_true: np.ndarray, y_pred: np.ndarray) -> (
        np.ndarray, pd.DataFrame):
    """
    Align KMeans cluster labels with true class labels using the
    Hungarian algorithm.
    :param y_true: True class labels (can be strings or numbers).
    :type y_true: np.ndarray
    :param y_pred: Predicted cluster labels from KMeans.
    :type y_pred: np.ndarray
    :return:  Cluster predictions relabeled to best match true class
    labels and DataFrame showing how each cluster was mapped to a true class
    :rtype: (np.ndarray, pd.DataFrame)
    """

    # Ensure numpy arrays
    y_true: np.ndarray = np.array(y_true)
    y_pred: np.ndarray = np.array(y_pred)

    # Build contingency (confusion) matrix
    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create mapping from cluster → class
    label_mapping: dict = {col: list(np.unique(y_true))[row]
                           for row, col in zip(row_ind, col_ind)}

    # Apply mapping
    y_pred_aligned: np.ndarray = np.array([label_mapping[c] for c in y_pred])

    # Return readable mapping as DataFrame
    mapping_df: pd.DataFrame = pd.DataFrame(
        list(label_mapping.items()), columns=["Cluster", "Mapped_Class"]
    )

    return y_pred_aligned, mapping_df


# 1. Open the CSV file
print("1. We open the dataset")
df: pd.DataFrame = pd.read_csv('Meteorite_Landings.csv')

print("2. We preprocess the data and clean it")
# 2. Data preprocessing
# 2.1 Drop rows with empty values
df.dropna(inplace=True)

# 2.2 Transform the column fall_flag to numeric
df['fall_flag'] = [0 if x == "Fell" else 1 for x in df['fall']]

# 2.3 Separate the coordinates in Geolocation into two numeric fields
# 2.3.a First, clean the field, remove the () and split by ,
df['GeoLocation_clean'] = \
    df['GeoLocation'].str.replace("(", "").str.replace(")", "").str.split(",")
# 2.3.b Second create a new column for each value in the array created
# in each row above
df["Latitude"] = [x[0] for x in df["GeoLocation_clean"]]
df["Longitude"] = [x[1] for x in df["GeoLocation_clean"]]

# 2.4 We have to convert the column recclass that has multiple possible
# values into numeric. In order to do so, we will create a new
# column for each value and assign 0 if that row did not have that
# value or 1 if it did
recclass_encoded = pd.get_dummies(df['recclass'], prefix='class')
df = pd.concat([df, recclass_encoded], axis=1)

# 2.5 Convert mass_class to 3 different groups
mass_min: float = df['mass (g)'].min() - 1
mass_max: float = df['mass (g)'].max() + 1
df['mass_class']: pd.DataFrame = pd.cut(df['mass (g)'],
                                        bins=[mass_min, 1000, 10000,
                                              mass_max],
                                        labels=["Small", "Medium", "Large"])

# 2.6 Leave only the columns we need in the dataframe
df_clean: pd.DataFrame = df.drop(columns=["name", "id", "recclass",
                                          "reclat", "reclong", "GeoLocation",
                                          "mass (g)", "fall", "nametype",
                                          "GeoLocation_clean"])

# 2.7 Create independent variable x with the corresponding columns
df_x: pd.DataFrame = df_clean.drop(columns=["mass_class"])
print(df_x.columns)
# 2.8 Create dependent variable y with the corresponding column
df_y: pd.Series = df_clean["mass_class"]

print("2.5 Applying Standard Scaler to variable X")
# 7. Let's apply the standard scaler to x
std_scaler: StandardScaler = StandardScaler()
std_scaler.fit(df_x)
df_x_std: pd.DataFrame = std_scaler.transform(df_x)
print(df_x_std)

print("2.6 Diving our variables x and y into training and testing sets")
# 8. Let's split between training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_x_std, df_y,
                                                    test_size=0.3,
                                                    random_state=4)

print("2.7 Finally, let's establish y classes as integer labels")
classes_labels: {int: str} = {
    k: v for k, v in zip(range(0, len(np.unique(y_test))), np.unique(y_test))}
print(classes_labels)

"""
KMeans
"""
# 9.1 We create a base Logistic model and then create a OneVSOne
# classifier whose input is the Logistic model
print("3a. Building the Kmeans object with n_clusters = 3")
decision_kmeans_model: KMeans = KMeans(
    init="k-means++", n_clusters=len(df_y.unique()), n_init=12)

print("3a.1 Translating string labels y to numeric labels so "
      "predictions are numeric")
# Invert your dictionary (str → int)
label_to_int = {v: k for k, v in classes_labels.items()}
# Translate df_y (which contains strings) to numeric values
y_true_numeric: np.ndarray = np.array([label_to_int[label] for label in df_y])

# 10.1 We train the model using training variables
print("3a.1 Training the model with full variable x (unsupervised)")
decision_kmeans_model.fit(df_x_std)

# 10.1 We use x_test to predict values
print("3a.2 Predicting the labels of variable x")
y_pred_kmeans: np.ndarray = decision_kmeans_model.labels_

# 10.2 We need to translate the predictions to the original cluster
y_pred_kmeans, df_translation = align_kmeans_labels(y_true_numeric,
                                                    y_pred_kmeans)

print("3b Building the KNearest Neighbor object")
knn_model: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=14)
print("3b.1 Training the model with our training sets of x and y")
knn_model.fit(x_train, y_train)

print("3b.2 Using the model to predict values in the test set")
y_pred_knn: np.ndarray = knn_model.predict(x_test)

print("4. Comparing accuracy scores")
accuracy_score_kmeans: float = accuracy_score(y_true_numeric, y_pred_kmeans)
accuracy_score_knn: float = accuracy_score(y_test, y_pred_knn)
print("Accuracy level KMeans Model = {0}".format(
    accuracy_score_kmeans))
print("Accuracy level KNearest Neighbor = "
      "{0}".format(accuracy_score_knn))
if accuracy_score_kmeans > accuracy_score_knn:
    print("Kmeans model was more accurate in this occasion")
elif accuracy_score_kmeans == accuracy_score_knn:
    print("Both have the same level of accuracy")
else:
    print("KNearest Neighbour model was more accurate in this occasion")

print("5. Getting Classification Reports")
classification_report_kmeans: str = classification_report(y_true_numeric,
                                                          y_pred_kmeans)
classification_report_knn: str = classification_report(y_test, y_pred_knn)
print("Classification report KMeans")
print(classification_report_kmeans)
print("Classification report KNearest Neighbor")
print(classification_report_knn)

print("6. Comparing precision by class")
precision_scores_kmeans: np.ndarray = \
    precision_score(y_true_numeric, y_pred_kmeans, average=None)
precision_scores_knn: np.ndarray = \
    precision_score(y_test, y_pred_knn, average=None)
print("6.1 Going class by class")
for i in range(0, len(precision_scores_kmeans)):
    print("Class = {0}".format(classes_labels[i]))
    print("Kmeans = {0}".format(
        precision_scores_kmeans[i]))
    print("KNN = {0}".format(precision_scores_knn[i]))
    if precision_scores_kmeans[i] > precision_scores_knn[i]:
        print("For class = {0} Kmeans is more precise".format(
            classes_labels[i]))
    elif precision_scores_kmeans[i] == precision_scores_knn[i]:
        print("For class = {0} both are equally "
              "precise".format(classes_labels[i]))
    else:
        print("For class = {0} KNN is more precise".format(classes_labels[i]))
    print("That means of all objects classified as {0} more of them "
          "belonged to that class (ratio of them was "
          "better)".format(classes_labels[i]))
    print("")

print("7. Comparing Recall scores by class")
recall_score_kmeans: np.ndarray = recall_score(
    y_true_numeric, y_pred_kmeans, average=None)
recall_score_knn: np.ndarray = recall_score(y_test, y_pred_knn, average=None)

for i in range(0, len(recall_score_knn)):
    print("For class = {0}".format(classes_labels[i]))
    print("Recall score for KMeans = {0}".format(
        recall_score_kmeans[i]))
    print("Recall score for KNN = {0}".format(
        recall_score_knn[i]))
    if recall_score_kmeans[i] > recall_score_knn[i]:
        print("For class = {0} Kmeans has better "
              "recall".format(classes_labels[i]))
    elif recall_score_kmeans[i] == recall_score_knn[i]:
        print("For class = {0} both have the same "
              "recall".format(classes_labels[i]))
    else:
        print("For class = {0} KNN has better recall".format(
            classes_labels[i]))
    print("This means we identified a better ratio of all the objects that "
          "truly belonged to a class (for instance, we classified all "
          "the elements of a class as that class even if we "
          "misclassified other objects as that class"
          ")".format(classes_labels[i]))
    print("")

print("8. Confusion Matrix")
confusion_matrix_kmeans: np.ndarray = confusion_matrix(y_true_numeric,
                                                       y_pred_kmeans)
confusion_matrix_knn: np.ndarray = confusion_matrix(y_test, y_pred_knn)
print(confusion_matrix_kmeans)
print(confusion_matrix_knn)
confusion_matrix_kmeans_str = \
    confusion_matrix_kmeans.astype(int).astype(str)
confusion_matrix_knn_str = \
    confusion_matrix_knn.astype(int).astype(str)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix_kmeans, annot=True, cmap='Blues', fmt='d',
            ax=axes[0], xticklabels=list(classes_labels.values()),
            yticklabels=list(classes_labels.values()),
            annot_kws={'color': 'black'})

axes[0].set_title('Kmeans Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix_knn, annot=True, cmap='Blues', fmt='d',
            ax=axes[1], xticklabels=list(classes_labels.values()),
            yticklabels=list(classes_labels.values()),
            annot_kws={'color': 'black'}, )
axes[1].set_title('KNN Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

"""
The poor performance of Kmeans, especially:
- Low recall in Small and Large clusters
- High, almost perfect recall in Medium cluster
Displays a problem with Kmeans:
That pattern is a hallmark of KMeans being “pulled” toward the densest
 region in your feature space. As this method is unsupervised, it does 
 not know what the original labels are so it does not train to divide
 based on the characteristics of the feature on each label.
"""
