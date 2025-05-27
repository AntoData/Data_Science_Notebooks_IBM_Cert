import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Apply standard scaler
print("3. Applying standard scaler to x")
std_scaler: StandardScaler = StandardScaler()
std_scaler.fit(df_life_expectancy[[x_1_col, x_2_col]],
               df_life_expectancy[y_data_col])
x_scaled = std_scaler.transform(df_life_expectancy[[x_1_col, x_2_col]])
print("x independent variables scaled: ")
print(x_scaled)
print("")

# Creating and training model
print("4. Creating our linear model and training it")
linear_model: LinearRegression = LinearRegression()
print("4.1 Getting our cross validation score")
cv_score = cross_val_score(linear_model, x_scaled,
                           df_life_expectancy[y_data_col], cv=5)
print("R^2 score = {0}".format(cv_score))
print("R^2 mean score = {0} with STD = {1}".format(cv_score.mean(),
                                                   cv_score.std()))
print("")

print("5. Using cross validation to predict values")
y_pred = cross_val_predict(linear_model, x_scaled,
                           df_life_expectancy[y_data_col], cv=5)
print("5.1 Using it to get the R^2 score again")
r2_sc_pred = r2_score(df_life_expectancy[y_data_col], y_pred)
print(r2_sc_pred)
