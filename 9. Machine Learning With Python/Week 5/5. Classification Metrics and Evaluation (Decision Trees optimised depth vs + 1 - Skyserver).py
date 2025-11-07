import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, \
    precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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


# 0.a As except for ObjectClass all our columns should be numeric,
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


# 1. We open the dataset
print("1. Opening our dataset")
df_skyserver: pd.DataFrame = pd.read_csv('Skyserver_Star_Galaxy_QSO.csv')
print(df_skyserver.head())

print("2. Preprocessing")
print("2.1 Converting all columns to type float")
# 3. We convert columns in x to float / Data Preprocessing
df_skyserver = column_to_float(df_skyserver)
print(df_skyserver.head())

print("2.2 Checking column types")
# 4. Let's check now the types of the columns and the content
print(df_skyserver.dtypes)
print(df_skyserver.head())

print("2.3 Building variable X")
# 5. Let's compose variable x
df_x: pd.DataFrame = df_skyserver[['Redshift',
                                   'u', 'g', 'r', 'i', 'z',
                                   'err_u', 'err_g', 'err_r', 'err_i',
                                   'err_z']]
print(df_x)
print("2.4 Building variable y")
# 6. Let's get variable y
df_y: pd.DataFrame = df_skyserver['ObjectClass']
print(df_x)

print("2.5 Applying Standard Scaler to variable X")
# 7. Let's apply the standard scaler to x
std_scaler: StandardScaler = StandardScaler()
std_scaler.fit(df_x)
df_x_std: pd.DataFrame = std_scaler.transform(df_x)
print(df_x_std)

print("2.6 Diving our variables x and y into training and testing sets")
# 8. Let's split between training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_x_std, df_y,
                                                    test_size=0.3,
                                                    random_state=4)

"""
Decision Trees Optimised
"""
# 9.1 We create a base Logistic model and then create a OneVSOne
# classifier whose input is the Logistic model
print("3a. Building the Decision Tree Classifier with best max depth "
      "which is 6")
decision_tree_model_op: DecisionTreeClassifier = DecisionTreeClassifier(
    criterion="entropy", max_depth=6)

# 10.1 We train the model using training variables
print("3a.1 Training the model with training variables x and y")
decision_tree_model_op.fit(x_train, y_train)

# 11.1 We use x_test to predict values
print("3a.2 Predicting the labels of the test set of x")
y_pred_des_tree_op: np.ndarray = decision_tree_model_op.predict(x_test)

print("3b. Building the Decision Tree Classifier with not optimised max "
      "depth 3")
decision_tree_model_no: DecisionTreeClassifier = DecisionTreeClassifier(
    criterion="entropy", max_depth=3)
print("3b.1 Training the model with our training sets of x and y")
decision_tree_model_no.fit(x_train, y_train)

print("3b.2 Using the model to predict values in the test set")
y_pred_des_tree_no: np.ndarray = decision_tree_model_no.predict(x_test)

print("4. Comparing accuracy scores")
accuracy_score_des_tree_op: float = accuracy_score(y_test, y_pred_des_tree_op)
accuracy_score_des_tree_no: float = accuracy_score(y_test, y_pred_des_tree_no)
print("Accuracy level Decision Tree Optimised Classifier Model = {0}".format(
    accuracy_score_des_tree_op))
print("Accuracy level Decision Tree Not Optimised Classifier Model = "
      "{0}".format(accuracy_score_des_tree_no))
if accuracy_score_des_tree_op > accuracy_score_des_tree_no:
    print("Decision Tree Classifier Optimised model was more accurate "
          "in this occasion")
elif accuracy_score_des_tree_op == accuracy_score_des_tree_no:
    print("Both have the same level of accuracy")
else:
    print("Decision Tree Classifier Not Optimised model was more "
          "accurate in this occasion")

print("5. Getting Classification Reports")
classification_report_des_tree_op: str = classification_report(
    y_test, y_pred_des_tree_op)
classification_report_des_tree_no: str = classification_report(
    y_test, y_pred_des_tree_no)
print("Classification report Decision Tree Optimised Classifier")
print(classification_report_des_tree_op)
print("Classification report Decision Tree Not Optimised Classifier")
print(classification_report_des_tree_no)

print("6. Comparing precision by class")
# Setting index for the different classes
classes_labels: {int: str} = {
    k: v for k, v in zip(range(0, len(np.unique(y_test))), np.unique(y_test))}
print(classes_labels)
precision_scores_des_tree_op: np.ndarray = \
    precision_score(y_test, y_pred_des_tree_op, average=None)
precision_scores_des_tree_no: np.ndarray = \
    precision_score(y_test, y_pred_des_tree_no, average=None)
print("6.1 Going class by class")
for i in range(0, len(precision_scores_des_tree_op)):
    print("Class = {0}".format(classes_labels[i]))
    print("Decision Tree Optimised Classifier = {0}".format(
        precision_scores_des_tree_op[i]))
    print("Decision Tree Not Optimised Classifier = {0}".format(
        precision_scores_des_tree_no[i]))
    if precision_scores_des_tree_op[i] > precision_scores_des_tree_no[i]:
        print("For class = {0} Decision Tree Optimised Classifier is more "
              "precise".format(classes_labels[i]))
    elif precision_scores_des_tree_op[i] == precision_scores_des_tree_no[i]:
        print("For class = {0} both are equally "
              "precise".format(classes_labels[i]))
    else:
        print("For class = {0} Decision Tree Not Optimised Classifier is"
              " more precise".format(classes_labels[i]))
    print("That means of all objects classified as {0} more of them "
          "belonged to that class (ratio of them was "
          "better)".format(classes_labels[i]))
    print("")

print("7. Comparing Recall scores by class")
recall_score_des_tree_op: np.ndarray = recall_score(
    y_test, y_pred_des_tree_op, average=None)
recall_score_des_tree_no: np.ndarray = recall_score(
    y_test, y_pred_des_tree_no, average=None)

for i in range(0, len(recall_score_des_tree_no)):
    print("For class = {0}".format(classes_labels[i]))
    print("Recall score for Decision Tree Optimised Classifier = {0}".format(
        recall_score_des_tree_op[i]))
    print("Recall score for Decision Tree Not Optimised Classifier = "
          "{0}".format(recall_score_des_tree_no[i]))
    if recall_score_des_tree_op[i] > recall_score_des_tree_no[i]:
        print("For class = {0} Decision Tree Optimised Classifier has better "
              "recall".format(classes_labels[i]))
    elif recall_score_des_tree_op[i] == recall_score_des_tree_no[i]:
        print("For class = {0} both have the same "
              "recall".format(classes_labels[i]))
    else:
        print("For class = {0}  Decision Tree Not Optimised Classifier"
              " has better recall".format(classes_labels[i]))
    print("This means we identified a better ratio of all the objects that "
          "truly belonged to a class (for instance, we classified all "
          "the elements of a class as that class even if we "
          "misclassified other objects as that class"
          ")".format(classes_labels[i]))
    print("")

print("8. Confusion Matrix")
confusion_matrix_des_tree_op: np.ndarray = confusion_matrix(
    y_test, y_pred_des_tree_op)
confusion_matrix_des_tree_no: np.ndarray = confusion_matrix(
    y_test, y_pred_des_tree_no)
print(confusion_matrix_des_tree_op)
print(confusion_matrix_des_tree_no)
confusion_matrix_des_tree_op_str = \
    confusion_matrix_des_tree_op.astype(int).astype(str)
confusion_matrix_des_tree_no_str = \
    confusion_matrix_des_tree_no.astype(int).astype(str)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix_des_tree_op, annot=True, cmap='Blues', fmt='d',
            ax=axes[0], xticklabels=list(classes_labels.values()),
            yticklabels=list(classes_labels.values()),
            annot_kws={'color': 'black'})

axes[0].set_title('Decision Tree Optimised Classifier Testing '
                  'Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix_des_tree_no, annot=True, cmap='Blues', fmt='d',
            ax=axes[1], xticklabels=list(classes_labels.values()),
            yticklabels=list(classes_labels.values()),
            annot_kws={'color': 'black'}, )
axes[1].set_title('Decision Tree Not Optimised Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
