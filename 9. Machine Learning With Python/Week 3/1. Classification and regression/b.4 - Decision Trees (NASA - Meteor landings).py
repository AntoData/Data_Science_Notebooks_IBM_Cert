import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
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
df: pd.DataFrame = pd.read_csv('Meteorite_Landings.csv')

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

# 3. Apply the standard scaler to variable x
std_scl: StandardScaler = StandardScaler()
std_scl.fit(df_x, df_y)
df_x_scl: pd.DataFrame = std_scl.transform(df_x)

# 4. Divide the variables into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(df_x_scl, df_y)

# 5. Let's build the decision tree with criterion entropy and iterate
# through different max depth levels
best_max_depth: int = 0
max_accuracy_score: float = 0.0

for max_depth_ in range(4, 21):
    decision_tree_meteor: DecisionTreeClassifier = \
        DecisionTreeClassifier(criterion="entropy", max_depth=max_depth_)

    # 6. Let's train the model using our testing sets
    decision_tree_meteor.fit(x_train, y_train)

    # 7. Let's use our test set for variable x to get some predictions
    # using our model
    y_pred: pd.DataFrame = decision_tree_meteor.predict(x_test)

    # 8. Let's get our accuracy score for this max depth level
    acc_scr: float = accuracy_score(y_test, y_pred)
    print("Max depth = {0} - accuracy score = {1}".format(max_depth_, acc_scr))

    if acc_scr > max_accuracy_score:
        max_accuracy_score = acc_scr
        best_max_depth = max_depth_

print("Best model has a max depth = {0}".format(best_max_depth))
print("With accuracy score = {0}".format(max_accuracy_score))

# 9. Let's build now the decision tree with the best max depth
decision_tree_meteor_best: DecisionTreeClassifier = \
    DecisionTreeClassifier(criterion="entropy", max_depth=best_max_depth)

# 10. Let's train the model again
decision_tree_meteor_best.fit(x_train, y_train)

# 11. Let's plot the decision tree
plot_tree(decision_tree_meteor_best)
plt.show()

# 12. Let's get the thresholds for each node in the tree
# Access tree structure
tree = decision_tree_meteor_best.tree_

# Feature indices used at each node
feature_indices = tree.feature

# Thresholds at each node
thresholds = tree.threshold

# Classes distribution at each node
value = tree.value  # shape: (n_nodes, 1, n_classes)

# Print all thresholds and corresponding feature and class distribution
for i in range(tree.node_count):
    if feature_indices[i] != -2:  # -2 indicates a leaf
        print(f"Node {i}:")
        feature_i = df_x.columns[feature_indices[i]]
        print(f"  Feature: {feature_i}")
        print(f"  Threshold: {thresholds[i]}")
        print(f"  Class distribution: {value[i]}")
        print()
