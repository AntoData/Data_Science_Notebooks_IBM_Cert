import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

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

# 1. Open the CSV file
print("1. We open the dataset")
df: pd.DataFrame = pd.read_csv('Meteorite_Landings.csv')

print("2. We preprocess the data and clean it")
# 2. Data preprocessing
# 2.1 Drop rows with empty values for Geolocation and mass (g)
print("2.1 Dropping NA from target (mass (g)) and Geolocation (as it "
      "needs preprocessing)")
df.dropna(subset=["GeoLocation", "mass (g)"], inplace=True)

# 2.2 Separate the coordinates in Geolocation into two numeric fields
print("2.2 Preprocessing coordinates in Geolocation")
# 2.2.1 First, clean the field, remove the () and split by ,
print("2.2.1 Transforming GeoLocation into two numeric fields Latitude "
      "and Longitude")
df['GeoLocation_clean'] = \
    df['GeoLocation'].str.replace("(", "").str.replace(")", "").str.split(",")
print("2.2.2 Creating columns Latitude and Longitude")
# 2.2.2 Second create a new column for each value in the array created
# in each row above
df["Latitude"] = [x[0] for x in df["GeoLocation_clean"]]
df["Longitude"] = [x[1] for x in df["GeoLocation_clean"]]
print("2.3 Converting target mass (g) into 3 classes")
# 2.3 Convert mass_class to 3 different groups
mass_min: float = df['mass (g)'].min() - 1
mass_max: float = df['mass (g)'].max() + 1
df['mass_class']: pd.DataFrame = pd.cut(df['mass (g)'],
                                        bins=[mass_min, 1000, 10000,
                                              mass_max],
                                        labels=["Small", "Medium", "Large"])
print("2.4 Keeping only relevant columns")
# 2.4 Leave only the columns we need in the dataframe
df_clean: pd.DataFrame = df.drop(columns=["name", "id", "recclass",
                                          "reclat", "reclong", "GeoLocation",
                                          "mass (g)", "fall", "nametype",
                                          "GeoLocation_clean"])
print("2.5 Getting variable x")
# 2.5 Create independent variable x with the corresponding columns
df_x_pre: pd.DataFrame = df_clean.drop(columns=["mass_class"])
print(df_x_pre.columns)
print("2.6 Getting target y")
# 2.6 Create dependent variable y with the corresponding column
df_y_pre: pd.DataFrame = df_clean[["mass_class"]]

print("2.7 Getting only numeric features of train set to "
      "apply Standard Scaler later")
x_pre_numeric_columns: pd.Index = \
    df_x_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_pre_numeric_columns)
print("2.8 Getting non-numeric features, in this case all boolean features")
x_pre_non_numeric_cols: pd.Index = df_x_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
print(x_pre_non_numeric_cols)
print("2.9 Creating the pipeline for numeric features")
numerical_transformer: Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
print("2.10 Creating the pipeline for numeric features")
categorical_transformer: Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
print("2.11 Joining both to create a preprocessor for the final pipeline")
preprocessor: ColumnTransformer = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, x_pre_numeric_columns),
        ('cat', categorical_transformer, x_pre_numeric_columns)
    ])

print("3. Creating the pipeline")
pipeline_: Pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA()),
    ('classifier_knn', KNeighborsClassifier())
])

print("3.1 Pipeline parameters")
param_grid_: {str: [int]} = {'pca__n_components': [3, 4, 5],
                             'classifier_knn__n_neighbors':
                                 [i for i in range(45, 70)]
                             }
print(param_grid_)

print("4. Splitting our variables x and y into training and testing sets")
x_train: pd.DataFrame
x_test: pd.DataFrame
y_train: pd.DataFrame
y_test: pd.DataFrame
x_train, x_test, y_train, y_test = train_test_split(df_x_pre, df_y_pre,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=df_y_pre)
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
grid_search_cv.fit(x_train, y_train.to_numpy().ravel())

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
