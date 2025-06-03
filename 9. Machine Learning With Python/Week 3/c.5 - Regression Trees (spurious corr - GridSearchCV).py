import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

reg_tree: DecisionTreeRegressor = DecisionTreeRegressor()

"""
5. We build the object GridSearchCV to get the best model possible
"""
parameters: dict = {"max_depth": [x for x in range(4, 21)]}
grid_search_cv: GridSearchCV = GridSearchCV(reg_tree, parameters,
                                            scoring="r2",
                                            cv=5)

"""
6. Let's train the model
"""
print("6. Let's train the model")
grid_search_cv.fit(df_x_scaled, df_y)

"""
7. Let's get the best possible model
"""
best_params: dict = grid_search_cv.best_params_
best_score: float = grid_search_cv.best_score_
print("Best model = {0}".format(best_params))
print("Best score = {0}".format(best_score))
