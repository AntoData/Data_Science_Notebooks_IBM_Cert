import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

# 9.1 We create a variable that contains the different possible values
# of parameter C to use GridSearch
param_grid = {
    'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# 10.1 We create a GridSearchCV object to get the best model
grid_search_cv: GridSearchCV = GridSearchCV(logistic_ovo, param_grid,
                            scoring="accuracy", cv=5)
# 11.1 We train the model
grid_search_cv.fit(df_x_std, df_y)

# 12.1 We get the best model
print("One vs One: Best parameter C = {0}".format(grid_search_cv.best_params_))
print("One vs One: Best accuracy score = {0}".format(
    grid_search_cv.best_score_))

# 13.1 We get probability predictions of belonging to classes
# One vs One does not support predict_proba
# y_pred_proba: pd.DataFrame = grid_search_cv.predict_proba(df_x_std)
# 14.1 We get log loss
# Therefore, we cannot get log_log
# log_l: float = log_loss(df_y, y_pred_proba)

"""
2. One vs All
"""
# 8.2 We build the one vs all model
# logistic_ova: LogisticRegression = LogisticRegression(multi_class='ovr')
logistic_ova: OneVsRestClassifier = OneVsRestClassifier(LogisticRegression())

# 9.2 We create a variable that contains the different possible values
# of parameter C to use GridSearch
param_grid = {
    'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# 10.2 We create a GridSearchCV object to get the best model
grid_search_cv_2: GridSearchCV = GridSearchCV(logistic_ova, param_grid,
                                                scoring="accuracy", cv=5)

# 11.2 We train the model
grid_search_cv_2.fit(df_x_std, df_y)

# 12.2 We get the best model possible
print("One vs All - Best parameter C = {0}".format(
    grid_search_cv_2.best_params_))
print("One vs All - Best accuracy score = {0}".format(
    grid_search_cv_2.best_score_))
# 13.2 We make predictions of probability
y_pred_proba_2: pd.DataFrame = grid_search_cv_2.predict_proba(df_x_std)

# 14.2 We use those predictions to get log loss
log_l2: float = log_loss(df_y, y_pred_proba_2)
print("One vs All - Log loss = {0}".format(log_l2))

"""
3. Multinomial
"""
# 8.3 We create the Logistic multinomial model
logistic_multi: LogisticRegression = LogisticRegression()

# 9.3 We create a variable that contains the different possible values
# of parameter C to use GridSearch
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# 10.3 We create a GridSearchCV object to get the best model
grid_search_cv_3: GridSearchCV = GridSearchCV(logistic_multi, param_grid,
                                              scoring="accuracy", cv=5)
# 11.3 We train the model
grid_search_cv_3.fit(df_x_std, df_y)
# 12.3 We get the best model
print("Multinomial - Best parameter C = {0}".format(
    grid_search_cv_3.best_params_))
print("Multinomial - Best accuracy score = {0}".format(
    grid_search_cv_3.best_score_))
# 13.3 We make some predictions of probability of belonging to classes
y_pred_proba_3: pd.DataFrame = grid_search_cv_3.predict_proba(df_x_std)
# 14.4 We use those predictions to get log loss
log_l3: float = log_loss(df_y, y_pred_proba_3)
print("Multinomial - Log loss = {0}".format(log_l3))