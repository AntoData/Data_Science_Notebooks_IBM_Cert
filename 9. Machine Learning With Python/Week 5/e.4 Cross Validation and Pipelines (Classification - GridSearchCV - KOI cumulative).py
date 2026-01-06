import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

"""
Data source:
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/
nph-tblView?app=ExoTbls&config=cumulative

Independent variables (x):
koi_period: Orbital Period [days]
koi_duration: Transit Duration [hrs] 
koi_depth: Transit Depth [ppm]
koi_prad:  Planetary Radius [Earth radii]
koi_teq: Equilibrium Temperature [K]
koi_insol: Insolation Flux [Earth flux]
koi_steff: Stellar Effective Temperature [K]
koi_srad: Stellar Radius [Solar radii]

Dependent variable: (y):
koi_disposition_y: Custom made variable made out of:
koi_disposition: Exoplanet Archive Disposition whose values can be:
- CANDIDATE
- CONFIRMED
- FALSE POSITIVE

We build koi_disposition_y like this:
- CONFIRMED -> 1
- FALSE POSITIVE -> 0
"""

print("1. We open the dataset")
df_koi: pd.DataFrame = pd.read_excel("cumulative_2025.03.19_12.51.51.xlsx")
df_koi.set_index(keys=["kepoi_name"], inplace=True)
print(df_koi)
print("")
print("2. We prepare the data for our model")
print("2.1 - We filter the candidate objects for later")
df_koi_candidates: pd.DataFrame = df_koi[
    df_koi["koi_disposition"] == "CANDIDATE"]
print("2.2 - We filter the whole dataset to only contain registers of "
      "confirmed or false positive exoplanets")
df_koi_features: pd.DataFrame = df_koi[
    df_koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
print("2.3 - We build a new column for variable y where CONFIRMED = 1 "
      "and FALSE POSITIVE = 0")
# We need to do this to avoid SettingWithCopyWarning
df_koi_features = df_koi_features.copy()
df_koi_features.loc[:, "koi_disposition_y"] = \
    df_koi_features["koi_disposition"].apply(
        lambda x: 1 if x == "CONFIRMED" else 0)
print("2.4 - We filter to get only the columns in variable x or y")
df_koi_features = df_koi_features[["koi_period", "koi_duration", "koi_depth",
                                   "koi_prad", "koi_teq", "koi_insol",
                                   "koi_steff", "koi_srad",
                                   "koi_disposition_y"]]
print("2.5 - We transform columns in x to numeric or turn them NA")
for col in ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
            "koi_insol", "koi_steff", "koi_srad"]:
    print("   - column = {0}".format(col))
    df_koi_features[col] = \
        pd.to_numeric(df_koi_features[col],
                      errors='coerce')
print("2.6 - We drop NA columns")
df_koi_features.dropna(inplace=True)
print(df_koi_features)
print("2.7 - We build now the variable x")
x_var: pd.DataFrame = df_koi_features[["koi_period", "koi_duration",
                                       "koi_depth", "koi_prad", "koi_teq",
                                       "koi_insol", "koi_steff", "koi_srad"]]
print("2.8 - We build now the variable y")
y_var = df_koi_features["koi_disposition_y"]
print("")

print("3. Creating the pipeline")
pipeline_: Pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier_knn', KNeighborsClassifier())
])

print("3.1 Pipeline parameters")
param_grid_: {str: [int]} = {'pca__n_components': [3, 4, 5],
                             'classifier_knn__n_neighbors':
                                 [4, 9, 14, 19, 24, 30, 35, 40, 45]
                             }
print(param_grid_)

print("4. Splitting our variables x and y into training and testing sets")
x_train: pd.DataFrame
x_test: pd.DataFrame
y_train: pd.DataFrame
y_test: pd.DataFrame
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y_var)
print("4.1 We need to use StratifiedKFold in order to use it later")
cv_: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=42)


print("Creating now our GridSearchCV to optimise the model")
grid_search_cv: GridSearchCV = GridSearchCV(
    estimator=pipeline_,
    param_grid=param_grid_,
    cv=cv_,
    scoring="accuracy",
    verbose=2
)

print("5. Training the model")
grid_search_cv.fit(x_train, y_train)

print("6. Best model parameters are: ")
print(grid_search_cv.best_params_)

print("7. Getting model's score")
test_score: float = grid_search_cv.score(x_test, y_test)
print("Model's score = {0}".format(test_score))

print("8. Using the model to predict x_test")
y_pred: pd.DataFrame = grid_search_cv.predict(x_test)

print("9. Confusion Matrix")
classes_labels: {int: str} = {
    k: v for k, v in zip(range(0, len(np.unique(y_test))), np.unique(y_test))}
confusion_matrix_knn: np.ndarray = confusion_matrix(y_test, y_pred)
print(confusion_matrix_knn)
confusion_matrix_svm_knn = \
    confusion_matrix_knn.astype(int).astype(str)
fig, axes = plt.subplots(1, 1, figsize=(12, 5))
sns.heatmap(confusion_matrix_knn, annot=True, cmap='Blues', fmt='d',
            ax=axes, xticklabels=list(classes_labels.values()),
            yticklabels=list(classes_labels.values()),
            annot_kws={'color': 'black'}, )
axes.set_title('KNN Testing Confusion Matrix')
axes.set_xlabel('Predicted')
axes.set_ylabel('Actual')
plt.tight_layout()
plt.show()

print("10. Getting precision by class")
precision_scores_knn: np.ndarray = \
    precision_score(y_test, y_pred, average=None)
print("10.1 Going class by class")
for i in range(0, len(precision_scores_knn)):
    print("Class = {0}".format(classes_labels[i]))
    print("KNN = {0}".format(precision_scores_knn[i]))
    print("Ratio of instances classified as {0} "
          "that truly belong to class {0}".format(classes_labels[i]))
    print("")

print("11. Getting Recall scores by class")
recall_score_knn: np.ndarray = recall_score(y_test, y_pred, average=None)
for i in range(0, len(recall_score_knn)):
    print("For class = {0}".format(classes_labels[i]))
    print("Recall score for KNN = {0}".format(
        recall_score_knn[i]))
    print("How many instances of class {0} there were and how many we "
          "found".format(classes_labels[i]))
    print("")
