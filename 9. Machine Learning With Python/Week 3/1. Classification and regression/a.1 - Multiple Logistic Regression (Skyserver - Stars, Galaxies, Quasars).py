import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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

"""
NOW WE WILL APPLY DIFFERENT RESOLUTION METHODS
1. One VS One
Trains a binary classifier for every pair of classes
During prediction all classifiers are used and a voting mechanism
decides the final class
"""
# 9.1 We create a base Logistic model and then create a OneVSOne
# classifier whose input is the Logistic model
logistic_model_base: LogisticRegression = LogisticRegression()
logistic_ovo = OneVsOneClassifier(logistic_model_base)

# 10.1 We train the model using training variables
logistic_ovo.fit(x_train, y_train)

# 11.1 We use x_test to predict values
y_pred: pd.DataFrame = logistic_ovo.predict(x_test)

# 12.1 We get the accuracy score now
acc_score: float = accuracy_score(y_test, y_pred)
print("Accuracy score = {0}".format(acc_score))

"""
2. One vs All
"""
# 9.2 We build the one vs all model
# logistic_ova: LogisticRegression = LogisticRegression(multi_class='ovr')
logistic_ova: OneVsRestClassifier = OneVsRestClassifier(LogisticRegression())

# 10.2 We train the model
logistic_ova.fit(x_train, y_train)

# 11.2 We use the x_test to predict value to test how well the method
# performs
y_pred: pd.DataFrame = logistic_ova.predict(x_test)

# 12.2 We use y_test and y_pred to get the accurary score to see
# how well the model performs
acc_score: float = accuracy_score(y_test, y_pred)
print("Accuracy score = {0}".format(acc_score))

"""
3. Multinomial
"""
# 9.3 We create the Logistic multinomial model
logistic_multi: LogisticRegression = LogisticRegression()

# 10.3 We train the model
logistic_multi.fit(x_train, y_train)

# 11.3 We use x_test with the model to predict values
y_pred: pd.DataFrame = logistic_multi.predict(x_test)

# 12.3 We use x_test with the model to predict probabilities
y_pred_proba: pd.DataFrame = logistic_multi.predict_proba(x_test)

# 13.3 We use the accuracy score to see how well this model is
# performing
acc_score: float = accuracy_score(y_test, y_pred)
print("Accuracy score = {0}".format(acc_score))

# 14.3 We use the log loss to check how well the model performs too
log_loss_score: float = log_loss(y_test, y_pred_proba)
print("Log loss = {0}".format(log_loss_score))