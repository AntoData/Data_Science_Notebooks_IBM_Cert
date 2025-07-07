import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

"""
SOURCE: 
#   VizieR Astronomical Server vizier.cds.unistra.fr
#    Date: 2025-05-12T15:26:56 [V7.4.6]
#   In case of problem, please report to:	cds-question@unistra.fr

Independent variable x:
B-V	Color:
index (difference between blue and visual magnitude)	
Strongest predictor of stellar temperature and thus spectral class.

Vmag:
Apparent visual magnitude	
Combined with parallax, it gives absolute magnitude (luminosity), 
which helps distinguish giants from dwarfs.

Plx:
Parallax (in milliarcseconds)
Used to estimate distance → essential for luminosity calculation.

Dependent variable y:
SpType:
Spectral type (e.g., “F5”, “K3V”, “G8III”)
This is the target of classification.
For modeling, you can extract just the spectral class letter ('F', 'K', 
etc.) as the label.
Astrophysical basis: Spectral type is a function of the star’s 
effective temperature, metallicity, and surface gravity, which 
correlate with observables like color index and luminosity.
"""
# 1. We open the dataset
df_stars: pd.DataFrame = pd.read_csv('vizier_stars_sampled_10000.tsv',
                                     delimiter="\t")
print(df_stars)
print(df_stars.dtypes)

# 2. Preprocessing
# 2.1 Drop NA
df_stars.dropna(inplace=True)
# 2.2 Sample 1000 records
df_stars = df_stars.sample(n=1000, random_state=42)
# 2.3 Getting only the first letter in the star type
df_stars['SpTypeSimple'] = df_stars['SpType'].str[0]
# No need for more preprocessing

# 3. We compose variable x and variable y
df_stars_x: pd.DataFrame = df_stars[['Vmag', 'Plx', 'B-V']]
df_stars_y: pd.DataFrame = df_stars['SpTypeSimple']

# 4. We apply the Standard Scaler
std_scl: StandardScaler = StandardScaler()
std_scl.fit(df_stars_x)
df_stars_x_std_scl: pd.DataFrame = std_scl.transform(df_stars_x)

# 5. Let's build the Decision Tree without max_depth as this will be
# optimised by Grid Search
decision_tree_stars: DecisionTreeClassifier = DecisionTreeClassifier(
        criterion="entropy")

# 6. Now we build the GridSearchCV object
max_depths: dict = {"max_depth": [x for x in range(3, 21)]}
grid_search_dec_tree_stars: GridSearchCV = GridSearchCV(decision_tree_stars,
                                                        scoring="accuracy",
                                                        param_grid=max_depths,
                                                        cv=5)

# 7. Let's train the model
grid_search_dec_tree_stars.fit(df_stars_x_std_scl, df_stars_y)

# 8. Let's get the best model
best_max_depth: int = grid_search_dec_tree_stars.best_params_["max_depth"]
max_accuracy_score: float = grid_search_dec_tree_stars.best_score_

print("")
print("Best max depth is {0}".format(best_max_depth))
print("With accuracy score = {0}".format(max_accuracy_score))

# 9. Let's build again the decision tree with the best model
decision_tree_stars_best: DecisionTreeClassifier = DecisionTreeClassifier(
    criterion="entropy", max_depth=best_max_depth
)

# 10. Let's train the model
decision_tree_stars_best.fit(df_stars_x_std_scl, df_stars_y)

# 11. Let's plot the decision tree
plot_tree(decision_tree_stars_best)
plt.show()

# 12. Let's get the threshold for the different features in the nodes
# of the tree
# Access tree structure
tree = decision_tree_stars_best.tree_

# Feature indices used at each node
feature_indices = tree.feature

# Thresholds at each node
thresholds = tree.threshold

# Classes distribution at each node
value = tree.value  # shape: (n_nodes, 1, n_classes)

# Print all thresholds and corresponding feature and class distribution
for i in range(tree.node_count):
    if feature_indices[i] != -2:  # -2 indicates a leaf
        print(f"Node {i}: ")
        print(f"  Feature: {df_stars_x.columns[feature_indices[i]]}")
        print(f"  Threshold: {thresholds[i]}")
        print(f"  Class distribution: {value[i]}")
        print()
