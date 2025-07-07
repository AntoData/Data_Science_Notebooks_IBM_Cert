import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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

# 5. Let's create the object for our decision tree whose criterion is
# entropy and let's iterate through different max depth levels
best_max_depth: int = 0
max_accuracy_score: float = 0

for max_depth_ in range(4, 20):
    decision_tree_stars: DecisionTreeClassifier = DecisionTreeClassifier(
        criterion="entropy", max_depth=max_depth_
    )

    # 6. We use cross_val_score to get the accuracy scores
    acc_scores: [float] = cross_val_score(decision_tree_stars,
                                          df_stars_x_std_scl, df_stars_y,
                                          scoring="accuracy", cv=5)

    # 7. We get the mean accuracy score
    acc_scr: float = acc_scores.mean()
    print("Max depth = {0} - accuracy score = {1}".format(max_depth_, acc_scr))

    if acc_scr > max_accuracy_score:
        max_accuracy_score = acc_scr
        best_max_depth = max_depth_

print("")
print("Best max depth is {0}".format(best_max_depth))
print("With accuracy score = {0}".format(max_accuracy_score))

# 8. Let's build again the decision tree with the best model
decision_tree_stars_best: DecisionTreeClassifier = DecisionTreeClassifier(
    criterion="entropy", max_depth=best_max_depth
)

# 9. Let's train the model
decision_tree_stars_best.fit(df_stars_x_std_scl, df_stars_y)

# 10. Let's plot the decision tree
plot_tree(decision_tree_stars_best)
plt.show()

# 11. Let's get the threshold for the different features in the nodes
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
