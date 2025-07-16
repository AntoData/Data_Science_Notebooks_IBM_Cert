import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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

print("4.a Building the Random Forest")
rf_meteor: RandomForestClassifier = RandomForestClassifier(random_state=41)
print("5.a Creating GridSearchCV object")
rf_param_grid: dict = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf: GridSearchCV = GridSearchCV(rf_meteor,
                                            param_grid=rf_param_grid, cv=5,
                                            scoring="accuracy")
print("6.a Training the Random Forest")
rf_training_start: float = time.time()
grid_search_rf.fit(df_x_std, df_y)
rf_training_end: float = time.time()
print("Training time = {0}".format(rf_training_end-rf_training_start))
print("Best score = {0}".format(grid_search_rf.best_score_))
print("Best params = {0}".format(grid_search_rf.best_params_))

print("4.b Building the XGBoost")
xgb_meteor: XGBClassifier = XGBClassifier(random_state=41)
print("5.b Creating GridSearchCV object")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search_xgb: GridSearchCV = GridSearchCV(xgb_meteor,
                                             param_grid=xgb_param_grid,
                                             cv=5, scoring="accuracy")
print("6.b Training the model")
classes_xgb: dict = {k_: v_ for v_, k_ in enumerate(df_y.unique())}
dy_numeric = df_y.apply(lambda y: classes_xgb[y])
xgb_training_start: float = time.time()
grid_search_xgb.fit(df_x_std, dy_numeric)
xgb_training_end: float = time.time()
print("Training time = {0}".format(xgb_training_end-xgb_training_start))
print("Best score = {0}".format(grid_search_xgb.best_score_))
print("Best params = {0}".format(grid_search_xgb.best_params_))