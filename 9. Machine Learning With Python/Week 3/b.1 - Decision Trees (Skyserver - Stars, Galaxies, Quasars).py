import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
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
# 1. As except for ObjectClass all our columns should be numeric,
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


# 2. We open the dataset
df_skyserver: pd.DataFrame = pd.read_csv('Skyserver_Star_Galaxy_QSO.csv')

# 3. We convert columns in x to float / Data Preprocessing
df_skyserver = column_to_float(df_skyserver)

# 4. Let's check now the types of the columns and the content
print(df_skyserver.dtypes)
print(df_skyserver.head())

# 5. Let's compose variable x
df_x: pd.DataFrame = df_skyserver[['Redshift',
                                   'u', 'g', 'r', 'i', 'z',
                                   'err_u', 'err_g', 'err_r', 'err_i',
                                   'err_z']]

# 6. Let's get variable y
df_y: pd.DataFrame = df_skyserver['ObjectClass']

# 7. Let's apply the standard scaler to x
std_scaler: StandardScaler = StandardScaler()
std_scaler.fit(df_x)
df_x_std: pd.DataFrame = std_scaler.transform(df_x)

# 8. Let's split between training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_x_std, df_y,
                                                    test_size=0.3,
                                                    random_state=4)

# 9. Let's create the decision tree whose criterion is entropy
# but we will iterate through a max depth between 4 and 10
max_accuracy_score: float = 0.0
best_max_depth: int = 0
for i in range(4, 20):
    decision_tree_object_class: DecisionTreeClassifier = \
        DecisionTreeClassifier(criterion="entropy", max_depth=10)

    # 10. Let's train the model using our training sets
    decision_tree_object_class.fit(x_train, y_train)

    # 11. Let's now get some predictions through this model using
    # our test set
    y_pred: pd.DataFrame = decision_tree_object_class.predict(x_test)

    # 12. Let's use those predictions and the testing set to get the
    # accuracy score of this depth
    acc_score: float = accuracy_score(y_test, y_pred)
    print("Max depth = {0} - Accuracy score = {1}".format(i, acc_score))
    if acc_score > max_accuracy_score:
        max_accuracy_score = acc_score
        best_max_depth = i

print("")
print("Best accuracy score was {0}".format(max_accuracy_score))
print("With max depth = {0}".format(best_max_depth))

# 13. Let's build the model with the best accuracy score
decision_tree_object_class: DecisionTreeClassifier = \
        DecisionTreeClassifier(criterion="entropy", max_depth=best_max_depth)

# 14. Let's train the model again
decision_tree_object_class.fit(x_train, y_train)

# 15. Let's plot the tree
plot_tree(decision_tree_object_class)
plt.show()

# 16. Finally, let's get the different thresholds for each node in the
# tree
# Access tree structure
tree = decision_tree_object_class.tree_

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
        print(f"  Feature: {df_x.columns[feature_indices[i]]}")
        print(f"  Threshold: {thresholds[i]}")
        print(f"  Class distribution: {value[i]}")
        print()
