"""
SOURCE:
https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/


Target (y):
- Rain_Event (binary, derived from PRCP if its value > 0)

Features (X):
- TEMP (Mean Temperature)
- DEWP (Mean Dew Point Temperature)
- SLP (Mean Sea Level Pressure)
- WDSP (Mean Wind Speed)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

print("1. Open dataset")
df_noaa_weather: pd.DataFrame = \
    pd.read_excel('aggregated_NOAA_weather_data.xlsx')
print(df_noaa_weather)
print("")

print("2. Extracting features x and y and preparing the dataset")
df_noaa_weather_features = df_noaa_weather[['TEMP', 'DEWP',
                                            'SLP', 'WDSP',
                                            'PRCP']].copy()
print("2.1 - Setting all columns to numeric and dropping non numeric values")
for col in df_noaa_weather_features:
    print("  - {0} to numeric".format(col))
    df_noaa_weather_features[col] = \
        pd.to_numeric(df_noaa_weather_features[col],
                      errors='coerce')

print("2.2 Dropping NA values")
df_noaa_weather_features.dropna(inplace=True)

print("2.3 Building target variable y from PRCP")
print("If PRCP > 0 y=1 else y=0")
df_noaa_weather_features['y_prcp'] = df_noaa_weather_features['PRCP'].apply(
    lambda x: 1 if float(x) > 0 else 0
).copy()
print("2.4 Getting features for x")
x_var = df_noaa_weather_features[['TEMP', 'DEWP', 'SLP', 'WDSP']].copy()
y_var = df_noaa_weather_features['y_prcp'].copy()
print("")

print("3. Applying standard scaler to x")
std_scaler = StandardScaler()
x_var_scaled = std_scaler.fit_transform(x_var)
print("")

print("4. Dividing dataset in train and test sets")
x_train, x_test, y_train, y_test = train_test_split(x_var_scaled, y_var)
print("")

print("5. Training logistic regression model using train datasets")
logistic_reg = LogisticRegression()
logistic_reg.fit(x_train, y_train)
print("")

print("6. Predicting probability using x_text")
y_pred_proba = logistic_reg.predict_proba(x_test)
print("")

print("7. Getting log loss score using y_test and our predictions")
log_loss_score = log_loss(y_test, y_pred_proba)
print(log_loss_score)
if log_loss_score < 0.5:
    print("Model is somewhat accurate")
else:
    print("Model is not accurate")