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
Source: https://skyserver.sdss.org/dr17/en/tools/search/sql.aspx
Query:
 SELECT TOP 100
  s.class         AS ObjectClass,   -- STAR, GALAXY, or QSO
  s.z             AS Redshift,
  p.psfMag_u      AS u,   p.psfMag_g   AS g,
  p.psfMag_r      AS r,   p.psfMag_i   AS i,
  p.psfMag_z      AS z,
  p.psfMagErr_u   AS err_u, p.psfMagErr_g AS err_g,
  p.psfMagErr_r   AS err_r, p.psfMagErr_i AS err_i,
  p.psfMagErr_z   AS err_z,
  sp.teffadop     AS Teff,
  sp.loggadop     AS logg,
  sp.fehadop      AS FeH
FROM SpecObjAll AS s
JOIN PhotoObjAll AS p
  ON s.bestObjID = p.objID
JOIN sppParams AS sp
  ON s.specObjID = sp.specObjID
WHERE s.class IN ('<VAR>')

Where VAR can be STAR, GALAXY or QSO
Dependent variable y
Column: ObjectClass	
Distinct astrophysical populations: stars vs. galaxies vs. quasars

Independent variable x
Columns:	
Redshift    Cosmological Doppler shift distinguishes Galactic vs. 
extragalactic objects
u, g, r, i, z	Broadband SED shapes differ by object type 
(stellar, galactic, AGN disk)
err_u, err_g, err_r, err_i, err_z	Measurement precision correlates 
with brightness and class identification
"""


# 0.a As except for ObjectClass all our columns should be numeric,
# we provide this function which will turn any column to float


def column_to_float(df: pd.DataFrame):
    """
    Converts every column but column ObjectClass to float

    :param df: Dataframe that contains variables x and y
    :return: Dataframe where columns that form variable x are float
    """
    for column_ in df.columns:
        if column_ != "ObjectClass":
            df[column_] = df[column_].str.replace('.', '')
            df[column_] = df[column_].astype('float64')
    return df


# 1. We open the dataset
print("1. Opening our dataset")
df_skyserver: pd.DataFrame = pd.read_csv('Skyserver_Star_Galaxy_QSO.csv')
print(df_skyserver.head())

print("2. Preprocessing")
print("2.1 Converting all columns to type float")
# 3. We convert columns in x to float / Data Preprocessing
df_skyserver = column_to_float(df_skyserver)
print(df_skyserver.head())

print("2.2 Checking column types")
# 4. Let's check now the types of the columns and the content
print(df_skyserver.dtypes)
print(df_skyserver.head())

print("2.3 Building variable X")
# 5. Let's compose variable x
df_x: pd.DataFrame = df_skyserver[['Redshift',
                                   'u', 'g', 'r', 'i', 'z',
                                   'err_u', 'err_g', 'err_r', 'err_i',
                                   'err_z']]
print(df_x)
print("2.4 Building variable y")
# 6. Let's get variable y
df_y: pd.DataFrame = df_skyserver['ObjectClass']
print(df_x)

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
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=df_y)
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
