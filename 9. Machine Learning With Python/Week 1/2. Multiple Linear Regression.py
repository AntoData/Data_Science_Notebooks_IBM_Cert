import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
std_scaler = StandardScaler()
std_scaler.fit(df_life_expectancy[[x_1_col, x_2_col]],
               df_life_expectancy[y_data_col])
x_scaled = std_scaler.transform(df_life_expectancy[[x_1_col, x_2_col]])
print("x independent variables scaled: ")
print(x_scaled)

print("4. Dividing the datasets into training and test sets")
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, df_life_expectancy[y_data_col], train_size=0.4)
print("5. Creating and training the multiple linear model")
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
print("6. Now we get the prediction of x_test")
y_pred = linear_model.predict(x_test)
print("7. We get now our metrics MSE and R^2")
mse = mean_squared_error(y_test, y_pred)
r2_sc = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE) = {0}".format(mse))
print("R^2 Score = {0}".format(r2_sc))
# It looks like predictions are not very accurate
