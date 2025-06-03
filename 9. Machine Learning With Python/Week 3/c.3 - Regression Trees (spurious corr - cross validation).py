import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
import scipy

"""
SOURCE: https://www.tylervigen.com/spurious/correlation/
1781_bachelors-degrees-awarded-in-psychology_correlates-with_the-
number-of-groundskeepers-in-utah
"""

"""
1. We need to open the dataset
"""
print("1. We open the dataset")
df_lr: pd.DataFrame = pd.read_csv('SimpleLinearR - Hoja 1.csv')
print(df_lr)
print("")
print("Let's arrange this dataset a little")
df_lr.set_index(keys=["Unnamed: 0"], inplace=True)
df_lr = df_lr.transpose()
df_lr = df_lr.reset_index()
df_lr.rename(columns={"index": "Year"}, inplace=True)
df_lr.set_index(keys=["Year"], inplace=True)
print(df_lr)
print("")

"""
2. Let's see if both variables are correlated
"""
print("2. Let's see if the variables are correlated")
print("Using dataframe's .corr()")
print("Correlation coefficient matrix")
print(df_lr.corr())
print("")
print("Pearson correlation coefficient and p-value using ")
pearson_coef, p_value = scipy.stats.pearsonr(df_lr[df_lr.columns[0]],
                                             df_lr[df_lr.columns[1]])
print("Pearson coefficient = {0}".format(pearson_coef))
print("p_value = {0}".format(p_value))
if abs(pearson_coef) >= 0.8:
    print("Strong correlation")
else:
    print("Not strong correlation")

if p_value < 0.001:
    print("Strong certainty")
else:
    print("Not strong certainty")
print("")


"""
3. Let's apply the standard scaler to x
"""
print("3. Let's apply the standard scaler to x")
df_x: pd.DataFrame = df_lr[df_lr.columns[0]]
df_y: pd.DataFrame = df_lr[df_lr.columns[1]]

scaler = StandardScaler()
# 1. First, we need to fit it
scaler.fit(df_x.to_frame())
# 2. Now we can use it to scale
df_x_scaled: pd.DataFrame = scaler.transform(df_x.to_frame())

"""
4. Now we create the linear model and train it
"""
print("4. Building the regression tree, we can iterate through several "
      "max depth")
best_max_depth: int = 0
best_score: float = 0.0
max_depth_score_map: dict = {}
for max_depth_ in range(1, 30):
    print("Max_depth ={0}".format(max_depth_))
    reg_tree: DecisionTreeRegressor = \
        DecisionTreeRegressor(criterion="squared_error",
                              max_depth=max_depth_,
                              random_state=42)

    """
    5. We work out now mean squared error (MSE) and r2 score (r^2)
    """
    print("5. We calculate MSE and R^2 score")
    print("5.1 We predict our test values")
    y_pred: pd.DataFrame = cross_val_predict(reg_tree, df_x_scaled, df_y, cv=5)
    print("")
    print("Getting MSE")
    mse: np.ndarray = mean_squared_error(df_y, y_pred)
    print("Max depth = {0} - Mean Squared Error: MSE = {1}".format(
        max_depth_, mse))
    r2_sc: np.ndarray = cross_val_score(reg_tree, df_x_scaled, df_y, cv=5)
    print("Max depth = {0} - R^2 Score (R2) = {1}".format(max_depth_, r2_sc))
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
