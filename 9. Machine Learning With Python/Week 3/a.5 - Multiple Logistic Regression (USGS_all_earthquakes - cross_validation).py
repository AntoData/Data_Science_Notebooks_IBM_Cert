import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, \
    cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import log_loss

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

"""
STRATEGY:
1. One vs All
Binary classifier for each class.
Binary classifier one vs all the other classes
"""
# 4.1 We build the logistic regression model
log_model: LogisticRegression = LogisticRegression()
# 5.1 We build the one vs all classifier and pass the logistic model
# object as an input
ova_model: OneVsRestClassifier = OneVsRestClassifier(log_model)
# 6.1 We use cross_val_score to get the accuracy scores
acc_scores_1 = cross_val_score(ova_model, df_scl, df_y,
                               scoring="accuracy", cv=5)
print("One Vs All: Accuracy scores = {0}".format(acc_scores_1))
print("One Vs All: Mean accuracy score = {0}".format(acc_scores_1.mean()))

"""
2. One vs One
Trains a binary classifier for every pair of classes
During prediction all classifiers are used and a voting mechanism
decides the final class
"""
# 4.2 We create the logistic model object
log_model_2: LogisticRegression = LogisticRegression()
# 5.2 We create the strategy One vs One classifier object and use the
# logistic model object as input
ovo_model: OneVsOneClassifier = OneVsOneClassifier(log_model_2)
# 6.2 We use cross_val_score to get the accuracy scores
acc_scores_2 = cross_val_score(ovo_model, df_scl, df_y, scoring="accuracy",
                               cv=5)
print("One vs One: Accuracy scores = {0}".format(acc_scores_2))
print("One vs One: Mean accuracy score = {0}".format(acc_scores_2.mean()))

"""
3. Multinomial default
"""
# 4.3 We build the logistic model
log_mn: LogisticRegression = LogisticRegression()

# 5.3 We use cross_val_score with scoring = accuracy to get the accuracy
# scores
acc_scores_3 = cross_val_score(log_mn, df_scl, df_y, scoring="accuracy",
                               cv=5)
print("Multinomial: Accuracy scores = {0}".format(acc_scores_3))
print("Multinomial: Mean accuracy score = {0}".format(acc_scores_3.mean()))
# 6.3 We use cross_val_predict with method predict_proba to make some
# predictions
y_pred_probas_3 = cross_val_predict(log_mn, df_scl, df_y,
                                    method="predict_proba", cv=5)
# 7.3 We use those predictions to get the log loss
log_loss_score_3: float = log_loss(df_y, y_pred_probas_3)
print("Multinomial: Log loss = {0}".format(log_loss_score_3))

