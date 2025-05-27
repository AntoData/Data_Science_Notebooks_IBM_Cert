import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

"""
SOURCE: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
"""
# 1. First we open the file and explore its data
print("1. Opening the file")
df_life_expectancy: pd.DataFrame = pd.read_csv('Life Expectancy Data.csv')
print("1.1 Dropping NA rows")
df_life_expectancy.dropna(inplace=True)
print("2. Getting correlation matrix")
corr_matrix = df_life_expectancy.corr(numeric_only=True)
print(corr_matrix)
print("")
print("2.1 Getting only correlation to variable Life expectancy")
y_data_col: str = "Life expectancy "
corr_life_expectancy = corr_matrix[[y_data_col]]
print(corr_life_expectancy)
print("")
print("2.2 Analysing the correlation data")
# First we remove the row that correlates Life expectancy with itself
corr_life_expectancy: pd.DataFrame = corr_life_expectancy.iloc[2:]
print("2.3 Getting highest correlation relationship")
highest_correlation: pd.Series = corr_life_expectancy.idxmax()
print(highest_correlation)
print("2.4 Getting second highest correlation relationship")
corr_life_expectancy = corr_life_expectancy.drop(
    index=[highest_correlation.iloc[0]])
highest_correlation2: pd.Series = corr_life_expectancy.idxmax()
print(highest_correlation2)
x_1_col: str = highest_correlation.iloc[0]
x_2_col: str = highest_correlation2.iloc[0]
print("So our independent variables are:")
print("- {0}".format(x_1_col))
print("- {0}".format(x_2_col))
print("")

# 3. Applying StandardScaler
print("3. Applying Standard Scaler to x")
std_scaler = StandardScaler()
std_scaler.fit(df_life_expectancy[[x_1_col, x_2_col]],
               df_life_expectancy[y_data_col])
x_scaled = std_scaler.transform(df_life_expectancy[[x_1_col, x_2_col]])
print("")

# 4. Applying polynomial transformation
print("4. Applying Polynomial Transformation to x with degree = 2")
polynomial_feat = PolynomialFeatures(degree=2)
x_scaled_p2 = polynomial_feat.fit_transform(x_scaled)
print("")

print("5. Training our Linear Model")
linear_model_p2 = LinearRegression()
linear_model_p2.fit(x_scaled_p2, df_life_expectancy[y_data_col])
print("")

print("6. Getting the prediction")
y_pred = linear_model_p2.predict(x_scaled_p2)
print("")

print("7. Getting Mean Squared Error and R^2 score")
mse = mean_squared_error(df_life_expectancy[y_data_col], y_pred)
r2_sc = r2_score(df_life_expectancy[y_data_col], y_pred)
print("Mean Squared Error (MSE) = {0}".format(mse))
print("R^2 score = {0}".format(r2_sc))
# We can see that now using a 2 degree polynomial linear correlation
# our predictions are better (correlation is higher)
