import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

"""
Data source:
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/
nph-tblView?app=ExoTbls&config=cumulative

Independent variables (x):
koi_period: Orbital Period [days]
koi_duration: Transit Duration [hrs] 
koi_depth: Transit Depth [ppm]
koi_prad:  Planetary Radius [Earth radii]
koi_teq: Equilibrium Temperature [K]
koi_insol: Insolation Flux [Earth flux]
koi_steff: Stellar Effective Temperature [K]
koi_srad: Stellar Radius [Solar radii]

Dependent variable: (y):
koi_disposition_y: Custom made variable made out of:
koi_disposition: Exoplanet Archive Disposition whose values can be:
- CANDIDATE
- CONFIRMED
- FALSE POSITIVE

We build koi_disposition_y like this:
- CONFIRMED -> 1
- FALSE POSITIVE -> 0
"""

print("1. We open the dataset")
df_koi: pd.DataFrame = pd.read_excel("cumulative_2025.03.19_12.51.51.xlsx")
df_koi.set_index(keys=["kepoi_name"], inplace=True)
print(df_koi)
print("")
print("2. We prepare the data for our model")
print("2.1 - We filter the candidate objects for later")
df_koi_candidates: pd.DataFrame = df_koi[
    df_koi["koi_disposition"] == "CANDIDATE"]
print("2.2 - We filter the whole dataset to only contain registers of "
      "confirmed or false positive exoplanets")
df_koi_features: pd.DataFrame = df_koi[
    df_koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
print("2.3 - We build a new column for variable y where CONFIRMED = 1 "
      "and FALSE POSITIVE = 0")
# We need to do this to avoid SettingWithCopyWarning
df_koi_features = df_koi_features.copy()
df_koi_features.loc[:, "koi_disposition_y"] = \
    df_koi_features["koi_disposition"].apply(
        lambda x: 1 if x == "CONFIRMED" else 0)
print("2.4 - We filter to get only the columns in variable x or y")
df_koi_features = df_koi_features[["koi_period", "koi_duration", "koi_depth",
                                   "koi_prad", "koi_teq", "koi_insol",
                                   "koi_steff", "koi_srad",
                                   "koi_disposition_y"]]
print("2.5 - We transform columns in x to numeric or turn them NA")
for col in ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
            "koi_insol", "koi_steff", "koi_srad"]:
    print("   - column = {0}".format(col))
    df_koi_features[col] = \
        pd.to_numeric(df_koi_features[col],
                      errors='coerce')
print("2.6 - We drop NA columns")
df_koi_features.dropna(inplace=True)
print(df_koi_features)
print("2.7 - We build now the variable x")
x_var: pd.DataFrame = df_koi_features[["koi_period", "koi_duration",
                                       "koi_depth", "koi_prad", "koi_teq",
                                       "koi_insol", "koi_steff", "koi_srad"]]
print("2.8 - We build now the variable y")
y_var = df_koi_features["koi_disposition_y"]
print("")

print("3. We apply the standard scaler")
scaler: StandardScaler = StandardScaler()
x_var_scaled = scaler.fit_transform(x_var, y_var)
print("")

print("4.a We create the Random Forest Classifier")
rf_mod: RandomForestClassifier = RandomForestClassifier(n_estimators=100,
                                                        random_state=41)
print("5.a We use cross val score to get the models scores")
rf_start_score: float = time.time()
rf_scores = cross_val_score(rf_mod, x_var_scaled, y_var, cv=5,
                            scoring="accuracy")
rt_end_score: float = time.time()
print("Scores Random Forest = {0}".format(rf_scores))
print("Average score Random Forest = {0}".format(rf_scores.mean()))
print("Cross val score time = {0}".format(rt_end_score-rf_start_score))
print("6.a Cross predicting")
rf_start_predict: float = time.time()
y_pred_rf = cross_val_predict(rf_mod, x_var_scaled, y_var, cv=5,
                              method="predict")
rf_end_predict: float = time.time()
print("Cross val predict time = {0}".format(rf_end_predict - rf_start_predict))

print("4.b We create the XGBoost")
xgb_mod: XGBClassifier = XGBClassifier(n_estimators=100,
                                       random_state=41)
print("5.b We use cross val score to get the models scores")
xgb_training_start: float = time.time()
xgb_scores = cross_val_score(xgb_mod, x_var_scaled, y_var, cv=5,
                             scoring="accuracy")
xgb_training_end: float = time.time()
print("Accuracy scores XGBoost = {0}".format(xgb_scores))
print("Mean accuracy scores XGBoost = {0}".format(xgb_scores.mean()))
print("Scoring time XGBoost = {0}".format(
    xgb_training_end-xgb_training_start))
print("6.b Cross val predicting")
xgb_predict_start: float = time.time()
y_pred_xgb = cross_val_predict(rf_mod, x_var_scaled, y_var, cv=5,
                               method="predict")
xgb_predict_end: float = time.time()
print("Cross val predict time = {0}".format(xgb_predict_end-xgb_predict_start))
