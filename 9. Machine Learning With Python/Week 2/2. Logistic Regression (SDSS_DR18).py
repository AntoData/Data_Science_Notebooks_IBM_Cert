import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

print("1. Opening dataset")
df_sdss: pd.DataFrame = pd.read_csv("SDSS_DR18.csv")
print(df_sdss)

print("2. Preprocessing data")
print("2.1 Selecting features")
features = ['u', 'g', 'r', 'i', 'z', 'redshift']
print("2.2 We only keep rows that are stars or galaxy")
df_sdss = df_sdss[df_sdss['class'].isin(['STAR', 'GALAXY'])]
print("2.3 We add a column that is 1 if entry is star or 0 otherwise")
df_sdss['star?'] = df_sdss['class'].apply(lambda x: 0 if x == 'STAR' else 1)
print("2.4 Finally preparing variables x and y")
x_var = df_sdss[features]
y_var = df_sdss['star?']
print("")

print("3. Splitting our data into train and test sets")
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var,
                                                    test_size=0.3)
print("")

print("4. Applying the standard scaler to x")
std_scaler: StandardScaler = StandardScaler()
x_train_norm = std_scaler.fit_transform(x_train.to_numpy())
x_test_norm = std_scaler.transform(x_test.to_numpy())
print("")

print("5. Training our Logistic Model")
log_model: LogisticRegression = LogisticRegression().fit(x_train_norm, y_train)
print("5.1 Predicting where entries x_test are stars (1) or galaxies (0)")
y_pred = log_model.predict(x_test_norm)
print("5.2 Getting the probability for our entries in x_test")
y_pred_proba = log_model.predict_proba(x_test_norm)
print("")

print("6. Evaluating the performance of our model using log_log")
model_perf = log_loss(y_test, y_pred_proba)
print(model_perf)