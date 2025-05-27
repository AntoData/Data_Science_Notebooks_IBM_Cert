import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss

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
    df_koi["koi_disposition"] == "CANDIDATE"].copy()
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

print("4. We create and train the Logistic Model using training sets")
logit_model = LogisticRegression()
print("")

print("5. We use cross val score with scoring = neg_log_loss")
log_loss_scores = cross_val_score(logit_model, x_var_scaled, y_var, cv=5,
                                  scoring='neg_log_loss')
print("scores = {0}".format(log_loss_scores))
mean_log_loss_score: float = abs(log_loss_scores.mean())
print("Mean score = {0}".format(mean_log_loss_score))
if mean_log_loss_score < 0.5:
    print("Model seems somewhat accurate")
else:
    print("Model is not accurate")
print("")

print("6. We use the cross_val_predict to get predictions for y")
y_preds = cross_val_predict(logit_model, x_var_scaled, y_var, cv=5,
                            method='predict_proba')
print("")

print("7. We use cross val predict to predict and get the log loss score "
      "again")
log_loss_score = log_loss(y_var, y_preds)
print(log_loss_score)
if log_loss_score < 0.5:
    print("The model is somewhat accurate")
else:
    print("The model is not accurate")
print("")

print("8. We train the model with the full dataset")
logit_model.fit(x_var_scaled, y_var)
print("")

print("9. We use the model to predict CANDIDATES")
for col in ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
            "koi_insol", "koi_steff", "koi_srad"]:
    print("   - column = {0}".format(col))
    df_koi_candidates[col] = \
        pd.to_numeric(df_koi_candidates[col],
                      errors='coerce')
df_koi_candidates = df_koi_candidates[["koi_period", "koi_duration",
                                       "koi_depth", "koi_prad", "koi_teq",
                                       "koi_insol", "koi_steff",
                                       "koi_srad"]].dropna(). \
    copy()
x_cand = df_koi_candidates[["koi_period", "koi_duration",
                            "koi_depth", "koi_prad", "koi_teq",
                            "koi_insol", "koi_steff", "koi_srad"]].copy()
y_pred = logit_model.predict(x_cand.to_numpy())
print("Predictions: ")
df_koi_candidates['y_prob'] = y_pred
print(df_koi_candidates)
