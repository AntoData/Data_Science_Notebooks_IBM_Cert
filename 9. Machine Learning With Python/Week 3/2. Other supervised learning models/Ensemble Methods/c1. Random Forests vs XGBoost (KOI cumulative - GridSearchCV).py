import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
rf_mod: RandomForestClassifier = RandomForestClassifier(random_state=41)
print("5.a We build the GridSearchCV object")
rf_param_grid: dict = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf: GridSearchCV = GridSearchCV(rf_mod, param_grid=rf_param_grid,
                                            cv=5, scoring="accuracy")
print("6.a Training the model")
rf_start: float = time.time()
grid_search_rf.fit(x_var_scaled, y_var)
rf_end: float = time.time()
print("Training time = {0}".format(rf_end - rf_start))
print("Best score = {0}".format(grid_search_rf.best_score_))
print("Best params = {0}".format(grid_search_rf.best_params_))

print("7.a Let's predict now the object that were labeled as CANDIDATE "
      "using our random forest")
x_cand = df_koi_candidates[["koi_period",
                            "koi_duration",
                            "koi_depth", "koi_prad",
                            "koi_teq",
                            "koi_insol", "koi_steff",
                            "koi_srad"]]
# We need to do this to avoid SettingWithCopyWarning
x_cand = x_cand.copy()
for col in ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
            "koi_insol", "koi_steff", "koi_srad"]:
    print("   - column = {0}".format(col))
    x_cand[col] = \
        pd.to_numeric(x_cand[col],
                      errors='coerce')
x_cand.dropna(inplace=True)
x_cand_scaled = scaler.fit_transform(x_cand)
rf_best_model: RandomForestClassifier = grid_search_rf.best_estimator_
y_pred_cand = rf_best_model.predict(x_cand_scaled)
x_cand["koi_disposition_y"] = y_pred_cand
print(x_cand)

print("4.b We create the XGBoost")
xgb_mod: XGBClassifier = XGBClassifier(random_state=41)
print("5.b Creating GridSearchCV object")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search_xgb: GridSearchCV = GridSearchCV(xgb_mod, cv=5, scoring="accuracy",
                                             param_grid=xgb_param_grid)

print("6.b Training the model")
xgb_start: float = time.time()
grid_search_xgb.fit(x_var_scaled, y_var)
xgb_end: float = time.time()
print("Training time = {0}".format(xgb_end-xgb_start))
print("Best score = {0}".format(grid_search_xgb.best_score_))
print("Best params = {0}".format(grid_search_xgb.best_params_))

print("7.b Let's predict now the object that were labeled as CANDIDATE "
      "using our random forest")
x_cand = df_koi_candidates[["koi_period",
                            "koi_duration",
                            "koi_depth", "koi_prad",
                            "koi_teq",
                            "koi_insol", "koi_steff",
                            "koi_srad"]]
# We need to do this to avoid SettingWithCopyWarning
x_cand = x_cand.copy()
for col in ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
            "koi_insol", "koi_steff", "koi_srad"]:
    print("   - column = {0}".format(col))
    x_cand[col] = \
        pd.to_numeric(x_cand[col],
                      errors='coerce')
x_cand.dropna(inplace=True)
x_cand_scaled = scaler.fit_transform(x_cand)
xgb_mod_best: XGBClassifier = grid_search_xgb.best_estimator_
y_pred_cand = xgb_mod_best.predict(x_cand_scaled)
x_cand["koi_disposition_y"] = y_pred_cand
print(x_cand)
