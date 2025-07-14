import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
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

print("5. Building the models with Ks from 1 to 200")
scores_by_k: dict = {}
best_score_by_k: dict = {}
deep_k: int = 200
acc = np.zeros(deep_k)
std_acc = np.zeros(deep_k)
for k in range(1, deep_k + 1):
    print("FOR K = {0}".format(k))
    knns_meteor: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k)
    print("5.1 Getting the scores using cross val score")
    scores: [float] = cross_val_score(knns_meteor, df_x_std, df_y, cv=5,
                                      scoring="accuracy")
    print("scores = {0}".format(scores))
    scores_by_k[k] = scores
    best_score_by_k[k] = max(scores)
    acc[k - 1] = max(scores)

best_k: int = max(best_score_by_k.items(), key=operator.itemgetter(1))[0]
print("Best K = {0}".format(best_k))
print("Accuracy score = {0}".format(best_score_by_k[best_k]))


plt.plot(range(1, deep_k+1), acc, 'g')
plt.fill_between(range(1, deep_k+1), acc - 1 * std_acc, acc + 1 * std_acc,
                 alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
