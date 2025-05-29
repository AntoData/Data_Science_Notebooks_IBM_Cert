import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

"""
SOURCE:
USGS “All Earthquakes, Past Month”
https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv

Independent variable X columns:
Variable	Physical Rationale
depth (km)	Seismic attenuation & energy reaching surface. 
            Deeper quakes lose more energy before reaching the surface, 
            so depth modulates felt intensity and thus the chance of 
            “significant” classification.
latitude, 
longitude	Tectonic/regional effects. Different fault systems and 
            plate‐boundary geometries produce different typical 
            magnitudes. Spatial coordinates serve as proxies for 
            those geologic settings.
gap (°)	    Data‐quality / detection geometry. “Gap” is the largest 
            azimuthal gap in station coverage around the event. 
            Large gaps often mean poorer‐constrained hypocenters—and 
            tend to occur for smaller or more remote events—introducing
             bias in who gets detected as “major.”

Dependent variable y:
mag:    Classified in the following 
        Minor - [0.0, 3.0) 
        Light - [3.0, 5.0)
        Moderate - [5.0, 7.0)
        Strong - [7.0, 10.0]
"""
# 1. We open the dataset
df: pd.DataFrame = pd.read_csv('USGS_all_earthquakes_past_month.csv')
# 2. Preprocessing data
# 2.1 Dropping rows with NA values
df.dropna(inplace=True, axis=0)
print(df)
# 2.2 Getting independent variable x
df_x = df[['depth','latitude', 'longitude','gap']]
# 2.3 Diving column mag in the classes described above for variable y
min_val = df['mag'].min() - 1e-6
max_val = df['mag'].max() + 1e-6
bins = [min_val, 3.0, 5.0, max_val]
df_y = pd.cut(df["mag"], labels=["Minor", "Light", "Moderate", "Strong"],
              bins=[min_val, 3.0, 5.0, 7.0, max_val])
print(df_y)

# 3. Applying Standard Scaler
std_scl: StandardScaler = StandardScaler()
std_scl.fit(df_x, df_y)
df_scl: pd.DataFrame = std_scl.transform(df_x)

# 4. Let's build the decision tree whose criterion is entropy
decision_tree_earthquakes: DecisionTreeClassifier = \
        DecisionTreeClassifier(criterion="entropy")

# 5. Let's build the GridSearchCV object
max_depths: dict = {"max_depth": [x for x in range(3, 21)]}
grid_search_dec_tree_eq: GridSearchCV = GridSearchCV(decision_tree_earthquakes,
                                                     param_grid=max_depths,
                                                     scoring="accuracy",
                                                     cv=5)

# 6. Let's train the model
grid_search_dec_tree_eq.fit(df_scl, df_y)

# 7. Now we get the best model possible
max_accuracy_score: float = grid_search_dec_tree_eq.best_score_
best_max_depth: int = grid_search_dec_tree_eq.best_params_["max_depth"]

print("Best max depth = {0}".format(best_max_depth))
print("Accuracy score = {0}".format(max_accuracy_score))

# 8. Let's use the best max depth to build a decision tree again
decision_tree_earthquakes_best: DecisionTreeClassifier = \
    DecisionTreeClassifier(criterion="entropy", max_depth=best_max_depth)

# 9. Let's train the model
decision_tree_earthquakes_best.fit(df_scl, df_y)

# 10. Let's plot the decision tree
plot_tree(decision_tree_earthquakes_best)
plt.show()

# 11. Let's get the thresholds for each node
# Access tree structure
tree = decision_tree_earthquakes_best.tree_

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
