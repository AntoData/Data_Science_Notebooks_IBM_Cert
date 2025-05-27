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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

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

print("3. Creating logistic regression model")
logistic_reg = LogisticRegression()
print("")

print("4. Building C parameters")
# Define the parameter grid (C controls regularization strength)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Smaller C → Stronger regularization
    'penalty': ['l2']  # L2 regularization (ridge-like behavior)
}
print("")

print("5. Training Grid Search CV model")
grid_search_obj = GridSearchCV(logistic_reg, param_grid, cv=5,
                               scoring='accuracy')
grid_search_obj.fit(x_var, y_var)
print("")

print("6. Best estimator and best score")
print(grid_search_obj.best_estimator_)
print(grid_search_obj.best_score_)