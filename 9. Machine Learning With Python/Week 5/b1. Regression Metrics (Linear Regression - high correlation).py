import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    root_mean_squared_error, r2_score
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
pearson_coef: float
p_value: float
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
x_train: pd.DataFrame
x_test: pd.DataFrame
y_train: pd.Series
y_test: pd.Series
x_train, x_test, y_train, y_test = train_test_split(df_lr[df_lr.columns[0]],
                                                    df_lr[df_lr.columns[1]],
                                                    test_size=0.4,
                                                    random_state=42)

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
4. Now we create the linear model and train it
"""
print("4. Creating linear model")
linear_model: LinearRegression = LinearRegression()

print("4.1 We train the model")
linear_model.fit(x_train.to_frame(), y_train)

"""
5. We work out now mean squared error (MSE) and r2 score (r^2)
"""
print("5. We calculate MSE and R^2 score")
print("5.1 We predict our test values: x_test")
y_pred: np.ndarray = linear_model.predict(x_test.to_frame())
print("")
print("Getting MAE")
mae: float = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: MAE = {0} groundskeepers".format(mae))
print("Getting MSE")
mse: float = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: MSE = {0}".format(mse))
print("Getting RMSE")
rmse: float = root_mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error: RMSE = {0} groundskeepers".format(rmse))
r2_sc: float = r2_score(y_test, y_pred)
print("R^2 Score (R2) = {0}".format(r2_sc))
if r2_sc > 0.8:
    print("Our predictions are accurate")
else:
    print("Error is too high")

"""
6. Let's represent our regression model our the points
"""
print("Scatter plot, regression vs original points")
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Groundskeepers: Actual vs Predicted (Linear Regression)")
plt.show()

residuals = (y_test - y_pred)
plt.hist(residuals, bins=5, color='lightblue', edgecolor='black')
plt.title('Groundskeepers Prediction Residuals')
plt.xlabel('Number of Groundskeepers')
plt.ylabel('Frequency')
plt.show()
print('Average error = ' + str(int(np.mean(residuals))))
print('Standard deviation of error = ' + str(int(np.std(residuals))))

# Create a DataFrame to make sorting easy
residuals_df = pd.DataFrame({
    'Actual': y_test,
    'Residuals': residuals
})

# Sort the DataFrame by the actual target values
residuals_df = residuals_df.sort_values(by='Actual')

# Plot the residuals
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'],
            marker='o', alpha=0.4,ec='k')
plt.title('Groundskeeper Prediciton Residuals Ordered by Actual '
          'number of groundskeepers')
plt.xlabel('Actual Values (Sorted)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
