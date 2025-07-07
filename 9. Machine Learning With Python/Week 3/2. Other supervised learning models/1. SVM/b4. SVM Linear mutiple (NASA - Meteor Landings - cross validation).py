import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

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
df_y = df_clean["mass_class"]

print("3. Data normalization -> Standard Scaler")
df_x_std: pd.DataFrame = StandardScaler().fit_transform(df_x, df_y)
print("4. We create the SVM model")
lin_svm = LinearSVC(
    class_weight='balanced', random_state=31, loss="squared_hinge",
    fit_intercept=True, max_iter=5000, multi_class="ovr")
print("4.b We wrap our method in a CalibratedClassifier")
cab_svm = CalibratedClassifierCV(lin_svm)

print("5. We use cross val score with scoring accuracy")
auc_roc_scs: [float] = cross_val_score(cab_svm, df_x_std, df_y,
                                       scoring="accuracy", cv=5)
print("Roc Auc Scores = {0}".format(auc_roc_scs))

print("6. Using cross val predict with method predict_proba "
      "to predict values in the test set")
y_pred = cross_val_predict(cab_svm, df_x_std, df_y,
                           method="predict_proba", cv=5)

print("7. Using predictions and test sets to get accuracy of the model")
accuracy_sc: float = roc_auc_score(df_y, y_pred, multi_class='ovr')
print("Accuracy score: {0}".format(accuracy_sc))
