import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

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
y_var: pd.DataFrame = df_life_expectancy[y_data_col]
print(y_var)

print("4. Creating and training the multiple regression tree and "
      "iterating through max_depths")
best_max_depth: int = 0
best_score: float = 0.0
max_depth_score_map: dict = {}
for max_depth_ in range(1, 21):
    print("Max depth = {0}".format(max_depth_))
    reg_tree: DecisionTreeRegressor = DecisionTreeRegressor(
        max_depth=max_depth_,
        random_state=42,
        criterion="squared_error"
    )
    print("5. Now we get the prediction using x")
    y_pred = cross_val_predict(reg_tree, x_scaled, y_var, cv=5)
    print("6. We get now our metrics MSE and R^2")
    mse: np.ndarray = mean_squared_error(y_var, y_pred)
    r2_sc: np. ndarray = cross_val_score(reg_tree,x_scaled, y_var, cv=5)
    print("Max depth = {0} - Mean Squared Error (MSE) = {1}".format(max_depth_,
                                                                    mse))
    print("Max depth = {0} - R^2 Score = {1}".format(max_depth_, r2_sc))

    max_depth_score_map[max_depth_] = r2_sc

    if best_score < r2_sc.mean():
        best_score = r2_sc.mean()
        best_max_depth = max_depth_
    elif max_depth_ == 1:
        best_score = r2_sc.mean()
        best_max_depth = max_depth_

for k_ in max_depth_score_map.keys():
    print("{0} - score = {1}". format(k_, max_depth_score_map[k_]))

print("")
print("Best model: max_depth = {0} - score = {1}".format(best_max_depth,
                                                         best_score))
