import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
3. We divide the dataset into train and test sets
"""
print("3. Dividing sets in train and test sets")
x_train, x_test, y_train, y_test = train_test_split(df_lr[df_lr.columns[0]],
                                                    df_lr[df_lr.columns[1]],
                                                    test_size=0.7)

print("x_train")
print(x_train)
print("")
print("x_test")
print(x_test)
print("")
print("y_train")
print(y_train)
print("")
print("y_test")
print(y_test)
print("")

"""
4. Now we apply the standard scaler to x_train and x_test
"""
print("4. We apply now the standard scaler to both x sets")
scaler = StandardScaler()
# 1. First, we need to fit it
scaler.fit(x_train.to_frame(), y_train)
# 2. Now we can use it to scale
x_train_scaled = scaler.transform(x_train.to_frame())
x_test_scaled = scaler.transform(x_test.to_frame())
print("x_train_scaled")
print(x_train_scaled)
print("")
print("x_test_scaled")
print(x_test_scaled)
print("")

"""
5. Now we create the linear model and train it
"""
print("5. Building the regression tree, we can iterate through several "
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

    print("5.1 We train the model")
    # We train the model using x_train_scaled and y_train
    reg_tree.fit(x_train_scaled, y_train)

    """
    6. We work out now mean squared error (MSE) and r2 score (r^2)
    """
    print("6. We calculate MSE and R^2 score")
    print("6.1 We predict our test values: x_test_scaled")
    y_pred = reg_tree.predict(x_test_scaled)
    print("")
    print("Getting MSE")
    mse = mean_squared_error(y_test, y_pred)
    print("Max depth = {0} - Mean Squared Error: MSE = {1}".format(
        max_depth_, mse))
    r2_sc = reg_tree.score(x_test_scaled, y_test)
    print("Max depth = {0} - R^2 Score (R2) = {1}".format(max_depth_, r2_sc))
    max_depth_score_map[max_depth_] = r2_sc

    if best_score < r2_sc:
        best_score = r2_sc
        best_max_depth = max_depth_
    elif max_depth_ == 1:
        best_score = r2_sc
        best_max_depth = max_depth_

for k_ in max_depth_score_map.keys():
    print("{0} - score = {1}". format(k_, max_depth_score_map[k_]))

print("")
print("Best model: max_depth = {0} - score = {1}".format(best_max_depth,
                                                         best_score))
