import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    root_mean_squared_error, r2_score

"""
PROBLEM: Is there linear correlation between the atmospheric CO₂ and 
Global Temperature Anomaly
Datasets:
1. NOAAGlobalTemp v6.0.0 (NCEI): 
SOURCE: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?
id=gov.noaa.ncdc:C01704&utm_source=chatgpt.com

- Monthly global surface temperature anomalies (°C)
- Coverage: 1850–2025

Columns / Variables:
       time : datetime64
           Monthly timestamp (Jan 1850 – May 2025)
       lat : float32
           Latitude coordinate in degrees north (-89.5 to +89.5)
       lon : float32
           Longitude coordinate in degrees east (0 to 359.5)
       z : float32
           Depth level (usually 0 for surface)
       anom : float32
           Surface temperature anomaly relative to 1971–2000 baseline 
           (°C)
       err : float32 (optional)
           Estimated uncertainty of anomaly (°C)

DEPENDENT VARIABLE Y: anom

2. Mauna Loa CO₂ Monthly Mean (NOAA GML)
SOURCE: https://gml.noaa.gov/ccgg/trends/data.html?utm_source

- Atmospheric CO₂ concentration (ppm)
- Coverage: 1958–2025

Columns:
       year : int
           Year of observation (1958 – present)
       month : int
           Month of observation (1–12)
       decimal date : float
           Year + fractional month (e.g., 1958.25 ≈ March 1958)
       average : float
           Monthly mean CO₂ concentration (ppm)
       deseasonalized : float
           CO₂ with seasonal cycle removed (ppm)
       ndays : int
           Number of daily means included (-1 = no data)
       sdev : float
           Standard deviation of daily means (ppm)
       unc : float
           Estimated uncertainty (ppm)
INDEPENDENT VARIABLE X: average
"""

# Path to your file
file_path_x_dataset: str = "co2_mm_mlo.csv"
file_path_y_dataset: str = \
    "NOAAGlobalTemp_v6.0.0_gridded_s185001_e202508_c20250909T092005.nc"

print("1. Opening files")
print("1.1 Opening file for y, nc file")
# Open the NetCDF dataset
original_df_y: xr.Dataset = xr.open_dataset(file_path_y_dataset)
original_df_y: pd.DataFrame = original_df_y.to_dataframe()

print("1.2 Opening file for x, csv file")
original_df_x: pd.DataFrame = pd.read_csv('co2_mm_mlo_clean.csv')

print("2. Preprocessing")
print("2.1 Reworking date format in dataset y")
print("Resetting index for time is a column")
original_df_y.reset_index(inplace=True)
print("Converting to DataTimeArray our column time")
original_df_y["time"] = pd.to_datetime(original_df_y["time"])
print("Getting year and month using dt and adding both as new columns")
original_df_y["year"] = original_df_y["time"].dt.year
original_df_y["month"] = original_df_y["time"].dt.month

print("2.2 Making sure both contain the same time period")

print("Getting lowest year each dataset contains")
year_min_y: int = min(original_df_y["year"].unique())
year_min_x: int = min(original_df_x["year"].unique())

print("Max of this pair of year is the year we should drop columns until")
year_cut: int = max(year_min_y, year_min_x)
print("Dropping the columns whose year is lower than the year where "
      "the most recently started dataset starts")
original_df_y.drop(original_df_y[original_df_y['year'].isin(range(
    year_min_y, year_cut))].index, inplace=True)
original_df_x.drop(original_df_x[original_df_x['year'].isin(range(
    year_min_x, year_cut))].index, inplace=True)

print("Now, we must get the months in the year where both datasets "
      "start that we should drop as they are not contained in both "
      "dataset, one of them starts later")
month_min_ds: int = min(original_df_y[original_df_y["year"] ==
                                      year_cut]["month"].unique())
month_min_df: int = min(original_df_x[original_df_x["year"] ==
                                      year_cut]["month"].unique())
month_cut: int = max(month_min_ds, month_min_df)

print("Dropping those rows that include months in the lowest year that "
      "are older than the first month of the most recently started dataset")
original_df_y.drop(original_df_y[(original_df_y['year'] == year_cut) & (
    original_df_y["month"].isin(range(month_min_ds, month_cut)))].index,
                   inplace=True)
original_df_x.drop(original_df_x[(original_df_x['year'] == year_cut) & (
    original_df_x["month"].isin(range(month_min_df, month_cut)))].index,
                   inplace=True)

print("2.3 Sorting datasets by year and then month")
original_df_y.sort_values(by=["year", "month"], inplace=True)
original_df_x.sort_values(by=["year", "month"], inplace=True)

print("2.4 Making datasets of the same length")
print("As time frame is the same now but number of observations is different,"
      "we will group by year and then month and get the means of x in "
      "y in each pair year and month")
df_y: pd.DataFrame = original_df_y.groupby(by=["year"])["anom"].mean()
df_x: pd.DataFrame = original_df_x.groupby(by=["year"])["average"].mean()

print("2.5 Asserting shapes are the same")
assert df_x.shape == df_y.shape

# We first divide into training and testing sets
print("2.6 We will divide the dataset in train and test sets")
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y)

print("2.7 Getting Pearson coefficient and p_value")
# We display the pearson correlation and p-value now
pearson_coef, p_value = stats.pearsonr(x_train, y_train)
print("Correlation Coef = {0}".format(pearson_coef))
print("p-value = {0}".format(p_value))
if abs(pearson_coef) < 0.8:
    print("Low correlation")
else:
    print("High correlation")
if p_value < 0.0001:
    print("Strong certainty")
else:
    print("Low certainty")

# Now we apply the Standard Scaler
print("2.8 - We fit the StandardScaler to x_train")
scaler: StandardScaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.to_frame())
x_test_scaled = scaler.transform(x_test.to_frame())

print("3. Creating the LinearRegression object")
# Now we create the Linear Model
linear_model: LinearRegression = LinearRegression()

# We train the model
print("3.1 Now we train the model using train y and scaled train x")
linear_model.fit(x_train_scaled, y_train)

print("Intercept: b0 = {0}".format(linear_model.intercept_))
print("Coef: b1 = {0}".format(linear_model.coef_))

print("3.2 We predict y for our scale test set of x")
y_pred = linear_model.predict(x_test_scaled)

"""
5. We work out now mean squared error (MSE) and r2 score (r^2)
"""
print("4. We calculate MAE, MSE, RMSE and R^2 score")
print("Getting MAE")
mae: float = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: MAE = {0} kt CO₂e".format(mae))
print("Getting MSE")
mse: float = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: MSE = {0}".format(mse))
print("Getting RMSE")
rmse: float = root_mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error: RMSE = {0} kt CO₂e".format(rmse))
r2_sc: float = r2_score(y_test, y_pred)
print("R^2 Score (R2) = {0}".format(r2_sc))
if r2_sc > 0.8:
    print("Our predictions are accurate")
else:
    print("Error is too high")

"""
6. Let's represent our regression model our the points
"""
print("5. Scatter plot, regression vs original points")
print("5. Scatter plot, predicted vs observed temperature anomaly")
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', lw=2)
plt.xlabel("Observed global temperature anomaly (°C)")
plt.ylabel("Predicted global temperature anomaly (°C)")
plt.title("Global temperature anomaly: observed vs predicted "
          "(linear regression)")
plt.tight_layout()
plt.show()

residuals = (y_test - y_pred)
plt.hist(residuals, bins=20, edgecolor='black')
plt.title('Residuals of predicted global temperature anomaly')
plt.xlabel('Residual (°C)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print('Average error = ' + str(float(np.mean(residuals))) + "ºC")
print('Standard deviation of error = ' + str(float(np.std(residuals))) + "ºC")

# Create a DataFrame to make sorting easy
residuals_df = pd.DataFrame({
    'Actual': y_test,
    'Residuals': residuals
})

# Sort the DataFrame by the actual target values
residuals_df = residuals_df.sort_values(by='Actual')

# Plot the residuals
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], marker='o',
            alpha=0.4, ec='k')
plt.title('Residuals vs observed global temperature anomaly')
plt.xlabel('Observed temperature anomaly (°C)')
plt.ylabel('Residual (°C)')
plt.axhline(0.0, color='k', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Convert test data to arrays for plotting
x_plot = x_test.values.reshape(-1, 1)
y_plot = y_test.values
y_line = linear_model.predict(x_test_scaled)  # predicted (already computed)

# Scatter: actual points
plt.figure(figsize=(8, 6))
plt.scatter(x_plot, y_plot, alpha=0.5, label="Observed data", color='blue')

# Regression line — sort by X to plot cleanly
sorted_idx = np.argsort(x_plot.flatten())
plt.plot(x_plot.flatten()[sorted_idx],
         y_line[sorted_idx],
         color='red', lw=2, label="Regression line")

plt.xlabel("Atmospheric CO₂ concentration (ppm)")
plt.ylabel("Global temperature anomaly (°C)")
plt.title("Global temperature anomaly vs CO₂ "
          "concentration\nwith linear regression fit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""
Our model performs pretty well after we got the averages yearly instead 
of per pair year-month. Look at R2 score:
0.93 show a high correlation between the CO2 at the temperature 
anomalies
"""