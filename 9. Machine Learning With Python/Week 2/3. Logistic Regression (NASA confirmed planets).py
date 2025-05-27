import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

"""
SOURCE: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/ç
nph-tblView?app=ExoTbls&config=q1_q17_dr25_sup_koi
"""

"""
VARIABLES IN X:
- koi_period	Orbital period (days)
- ra	Right Ascension (sky position coordinate)
- dec	Declination (sky position coordinate)
- koi_kepmag	Kepler magnitude (brightness of the star)

VARIABLE Y:
- koi_disposition: The category of this KOI from the Exoplanet 
Archive. Current values are CANDIDATE, 
FALSE POSITIVE, NOT DISPOSITIONED or CONFIRMED
However, we will only use FALSE POSITIVE and CONFIRMED

We will propose a logistic model in which the variables in x will 
determine if the object is a CONFIRMED exoplanet or was a FALSE 
POSITIVE

So we are trying to se if knowing the orbital period in days, 
temperature in the planet, transit depth, insolation influx (another
way to measure temperature), the sky-projected distance between the
 center of the stellar disc and the center of the planet disc at 
 conjunction and the time corresponding to the center
 of the first detected transit we can classify with confidence these 
 celestial object between CONFIRMED (1) exoplanets or FALSE POSITIVES
 (0, objects that seemed to be exoplanets but were not after further
 inspection)
"""

print("1. First we open the dataset")
df_conf_planets: pd.DataFrame = \
    pd.read_excel("q1_q17_dr25_sup_koi_2025.03.14_15.14.59.xlsx")
print(df_conf_planets.columns)
print("")

print("2. Data preparation")
print("2.1: We filter the column koi_disposition so we only retrieve"
      " rows with values CONFIRMED and FALSE POSITIVE which are the"
      " ones we need to propose the model")
df_conf_planets = \
    df_conf_planets[df_conf_planets['koi_disposition'].isin(
        ['CONFIRMED', 'FALSE POSITIVE'])]
print("2.2 We create a new column koi_disposition_y where CONFIRMED "
      "gets value 1 and FALSE POSITIVE gets value 0")
df_conf_planets['koi_disposition_y'] = df_conf_planets['koi_disposition']. \
    apply(lambda x: 1 if x == "CONFIRMED" else 0)
print("2.3 We filter the dataset so we only get the columns we need "
      "for the model (see documentation in the upper part of the file")
df_conf_planets = df_conf_planets[["koi_period", "ra", "dec", "koi_kepmag",
                                   "koi_disposition_y"]]
print(df_conf_planets.columns)
df_conf_planets.dropna(inplace=True)
print("2.4 Creating dataset for x")
x_var = df_conf_planets[["koi_period", "ra", "dec", "koi_kepmag"]]

print("2.5 Creating dataset for y")
y_var = df_conf_planets['koi_disposition_y']
print("")

print("3. Let's check if values 0 and 1 on y are balanced")
print("Number of rows = {0}".format(len(y_var)))
print("Number of 1s = {0}".format(len(y_var[y_var == 1])))
print("Number of 0s = {0}".format(len(y_var[y_var == 0])))
# It looks balanced
print("")

print("4. Applying Standard Scaler on x")
print(x_var)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_var, y_var)
print(x_scaled)
print("")

print("5. Diving our x dataset into training and testing sets")
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var)
print("")

print("6. Training our model")
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print("")

print("7. Predicting using our modal")
y_pred = log_reg.predict(x_test)
print("")

print("8. Evaluating model using log_log")
print("8.1 Predicting real probabilities")
y_pred_proba = log_reg.predict_proba(x_test)
log_loss_score = log_loss(y_test, y_pred_proba)
print(log_loss_score)
if log_loss_score > 0.5:
    print("This model is not accurate")