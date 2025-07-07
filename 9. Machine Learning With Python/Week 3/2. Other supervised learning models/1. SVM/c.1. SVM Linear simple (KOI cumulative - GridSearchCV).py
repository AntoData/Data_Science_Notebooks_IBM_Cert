import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

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

print("4. We create SVM model Model using training sets")
lin_svm: LinearSVC = LinearSVC(
    class_weight='balanced', random_state=31, loss="hinge",
    fit_intercept=False, max_iter=5000000)

print("5. We fine tune C and get the best estimator using GridSearch")
param_grid = {'C': [0.1, 1, 10, 100]}
grid_SVC_Linear: GridSearchCV = GridSearchCV(lin_svm, param_grid, cv=5)

print("6. We train the method")
grid_SVC_Linear.fit(x_var_scaled, y_var)

print("7. We get the best score possible")
best_score_svm: float = grid_SVC_Linear.best_score_
print("Best score = {0}".format(best_score_svm))

print("7. We get the best value for C")
best_c: float = grid_SVC_Linear.best_params_
print("Best C = {0}".format(best_c))

print("8. Let's predict now the object that were labeled as CANDIDATE "
      "using best C")
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
lin_svm: LinearSVC = LinearSVC(
    class_weight='balanced', random_state=31, loss="hinge",
    fit_intercept=False, max_iter=5000000, C=best_c["C"])
lin_svm.fit(x_var_scaled, y_var)
y_pred_cand = lin_svm.predict(x_cand_scaled)
x_cand["koi_disposition_y"] = y_pred_cand
print(x_cand)
