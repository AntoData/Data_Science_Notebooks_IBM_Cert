import matplotlib.pyplot as plt
import numpy as np
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
df_x = df[['depth', 'latitude', 'longitude', 'gap']]
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

# 4. Let's build the decision tree with criterion entropy
decision_tree_earthquakes: DecisionTreeClassifier = \
    DecisionTreeClassifier(criterion="entropy")

# 5. Let's build the Decision Tree without max_depth as this will be
# optimised by Grid Search
max_depths: dict = {"max_depth": [x for x in range(4, 21)]}
grid_search_dec_tree: GridSearchCV = GridSearchCV(decision_tree_earthquakes,
                                                  param_grid=max_depths,
                                                  scoring="accuracy",
                                                  cv=5,)
# 6. Let's train the model
grid_search_dec_tree.fit(df_scl, df_y)

# 7. Let's get the best model
best_max_depth: int = grid_search_dec_tree.best_params_["max_depth"]
max_accuracy_scores: float = grid_search_dec_tree.best_score_
print("Best model has max depth = {0}".format(best_max_depth))
print("With accuracy scores = {0}".format(max_accuracy_scores))

# 6. Let's build the decision tree again with the best max depth
decision_tree_earthquakes_best: DecisionTreeClassifier = \
    DecisionTreeClassifier(criterion="entropy", max_depth=best_max_depth)

# 7. Let's train the model
decision_tree_earthquakes_best.fit(df_scl, df_y)

# 8. Let's plot the tree
plot_tree(decision_tree_earthquakes_best)
plt.show()

# 9. Let's get the thresholds for each node in the tree
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
