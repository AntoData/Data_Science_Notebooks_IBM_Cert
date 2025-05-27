import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, \
    cross_val_predict
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

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

"""
NOW WE WILL APPLY DIFFERENT RESOLUTION METHODS
1. One VS One
Trains a binary classifier for every pair of classes
During prediction all classifiers are used and a voting mechanism
decides the final class
"""
# 8.1 We create a base Logistic model and then create a OneVSOne
# classifier whose input is the Logistic model
logistic_model_base: LogisticRegression = LogisticRegression()
logistic_ovo = OneVsOneClassifier(logistic_model_base)

# 9.1 We use cross validation scores with scoring = accuracy to get
# the accuracy scores of this model using 5 k-folds
acc_scores_1 = cross_val_score(logistic_ovo, df_x_std, df_y,
                               scoring="accuracy", cv=5)
print("One vs One: Accuracy scores = {0}".format(acc_scores_1))
print("One vs One: Mean accuracy score = {0}".format(acc_scores_1.mean()))

"""
2. One vs All
"""
# 8.2 We build the one vs all model
# logistic_ova: LogisticRegression = LogisticRegression(multi_class='ovr')
logistic_ova: OneVsRestClassifier = OneVsRestClassifier(LogisticRegression())

# 9.2 We use cross validation scores with scoring = accuracy to get
# the accuracy scores of this model using 5 k-folds
acc_scores_2 = cross_val_score(logistic_ova, df_x_std, df_y,
                               scoring="accuracy", cv=5)
print("One vs All: Accuracy scores = {0}".format(acc_scores_2))
print("One vs All: Mean accuracy score = {0}".format(acc_scores_2.mean()))

"""
3. Multinomial
"""
# 8.3 We create the Logistic multinomial model
logistic_multi: LogisticRegression = LogisticRegression()

# 9.3 We use cross validation scores with scoring = accuracy to get
# # the accuracy scores of this model using 5 k-folds
acc_scores_3 = cross_val_score(logistic_multi, df_x_std, df_y,
                               scoring="accuracy", cv=5)
print("Multinomial: Accuracy scores = {0}".format(acc_scores_3))
print("Multinomial: Mean accuracy scores = {0}".format(acc_scores_3.mean()))

# 10.3 We use cross_val_predict with method predict_proba to get some
# predictions to get log loss
y_pred_probas: pd.DataFrame = cross_val_predict(logistic_multi, df_x_std, df_y,
                                                method="predict_proba", cv=5)

# 11.3 We use the those predictions to get log loss
log_loss_score: float = log_loss(df_y, y_pred_probas)
print("Log loss = {0}".format(log_loss_score))
