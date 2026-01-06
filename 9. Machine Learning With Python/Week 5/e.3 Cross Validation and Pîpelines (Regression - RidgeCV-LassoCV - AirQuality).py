import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, \
    explained_variance_score, mean_absolute_error
from sklearn.linear_model import RidgeCV, LassoCV

"""
SOURCE: "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/
AirQualityUCI.zip"

Title:
“Field deployment of a multiparametric gas sensor array for air 
pollution monitoring purposes”

Authors:
Luis De Vito, Enrico Massera, Salvatore Pardo, Giuseppe Di Francia

Published In:
Sensors and Actuators B: Chemical, Volume 129, Issue 2, 2008, 
Pages 750–757

DOI:
10.1016/j.snb.2007.09.060

https://www.sciencedirect.com/science/article/abs/pii/S0925400507013693

This is an example about how to use RidgeCV and LassoCV with pipelines 
to find the best possible model

We will display their:
- Explained variance
- R^2 score
- Mean absolute error (MAE)
- Mean Square Error (MSE)
- Root Mean Square Error (RMSE)

We will split the dataset into training and testing sets 
and create 2 pipelines:
- Linear Ridge Regression (L2 Regularization) -> SimpleImputer, 
StdScaler, RidgeCV
- Linear Lasso Regression (L1 Regularization)  -> SimpleImputer, 
StdScaler, LassoCV

Then pick the alpha hyperparameter with the best accuracy score using 
pipelines
"""


def regression_results(y_true: np.ndarray | pd.DataFrame | pd.Series,
                       y_pred: np.ndarray | pd.DataFrame | pd.Series,
                       regr_type: str) -> (float, float, float, float):
    """
    Returns the explained variance, mean absolute error, mean squared
    error and R^2 scores for a certain variable y and its predictions

    :param y_true: Variable y
    :type y_true: A numpy array, dataframe or series
    :param y_pred: Predictions of variable y
    :type y_pred: A numpy array, dataframe or series
    :param regr_type: Type of regression applied (Simple, Ridge, Lasso
    only for printing purposes)
    :type regr_type: str
    :return: A tuple with explained variance, mean absolute error,
    mean squared error and R^2 score in that order
    :rtype: (float, float, float, float)
    """
    # Regression metrics
    ev: float = explained_variance_score(y_true, y_pred)
    mae: float = mean_absolute_error(y_true, y_pred)
    mse: float = mean_squared_error(y_true, y_pred)
    r2: float = r2_score(y_true, y_pred)

    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ', round(ev, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mae, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print()
    return ev, mae, mse, r2


def create_plot_scatter_actual_vs_predicted(
        axes_: plt.Axes, pos_x: int, pos_y: int,
        y_actual: np.ndarray | pd.DataFrame | pd.Series,
        y_pred: np.ndarray | pd.DataFrame | pd.Series,
        label: str, title: str) -> None:
    """
    Given the axes of a subplot, the position that the plot will occupy
    and the variables y and prediction of y, it adds a scatter plot to
    the figure

    :param axes_: Object that contains all the possible axes where we can
    add subplots
    :type axes_: plt.Axes (object we got from plt.subplots(...))
    :param pos_x: Position of this plot in the figure in axis x (column)
    :type pos_x: int
    :param pos_y: Position of this plot in the figure in axis y (row)
    :type pos_y: int
    :param y_actual: Variable y
    :type y_actual: Numpy array, Dataframe or Series
    :param y_pred: Prediction of variable y
    :type y_pred: Numpy array, Dataframe or Series
    :param label: Label to add to the scatter subplot
    :type label: str
    :param title: Title of the subplot
    :type title: str

    :return: None
    """

    axes_[pos_x, pos_y].scatter(y_actual, y_pred, color="red", label=label)
    axes_[pos_x, pos_y].plot([y_actual.min(),
                              y_actual.max()],
                             [y_actual.min(), y_actual.max()],
                             'k--')
    axes_[pos_x, pos_y].set_title(title)
    axes_[pos_x, pos_y].set_xlabel("Actual", )
    axes_[pos_x, pos_y].set_ylabel("Predicted", )


def create_plot_line_actual_vs_predicted(
        axes_: plt.Axes, pos_x: int, pos_y: int,
        y_actual: np.ndarray | pd.DataFrame | pd.Series,
        y_pred: np.ndarray | pd.DataFrame | pd.Series,
        label: str, title: str, line_colour: str) -> None:
    """
    Given the axes of a subplot, the position that the plot will occupy
    and the variables y and prediction of y, it adds a line of actual 
     y vs predicted y plot to the figure

    :param axes_: Object that contains all the possible axes where we can
    add subplots
    :type axes_: plt.Axes (object we got from plt.subplots(...))
    :param pos_x: Position of this plot in the figure in axis x (column)
    :type pos_x: int
    :param pos_y: Position of this plot in the figure in axis y (row)
    :type pos_y: int
    :param y_actual: Variable y
    :type y_actual: Numpy array, Dataframe or Series
    :param y_pred: Prediction of variable y
    :type y_pred: Numpy array, Dataframe or Series
    :param label: Label to add to the scatter subplot
    :type label: str
    :param title: Title of the subplot
    :type title: str
    :param line_colour: Colour of line to use (for example red)
    :type line_colour: str

    :return: None
    """

    axes_[pos_x, pos_y].plot(y_actual, label="Actual", lw=2)
    axes_[pos_x, pos_y].plot(y_pred, '--', lw=2, color=line_colour,
                             label=label)
    axes_[pos_x, pos_y].set_title(title)
    axes_[pos_x, pos_y].legend()


print("1. Let's open the file")
# Source of the dataset
file_path: str = "./AirQualityUCI.csv"
df_air_quality: pd.DataFrame = pd.read_csv(file_path, sep=";", decimal=",")
print("1.1 Let's check its content")
print(df_air_quality.head())

print("2. Data preprocessing")
df_air_quality_encoded: pd.DataFrame = df_air_quality.replace(-200, pd.NA)

print("2.1 Forcing columns to be numeric")
cols_to_fix: [str] = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
    "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
]

for col in cols_to_fix:
    df_air_quality_encoded[col] = pd.to_numeric(df_air_quality_encoded[col],
                                                errors='coerce')

print("2.2 Recheck now columns, types and data")
print(df_air_quality_encoded.columns)
print(df_air_quality_encoded.head())
print(df_air_quality_encoded.dtypes)

print("2.3 Getting training and testing sets")
df_x_pre: pd.DataFrame = df_air_quality_encoded[[
  "PT08.S1(CO)",       # Tin Oxide sensor
  "PT08.S2(NMHC)",     # Metal Oxide sensor
  "PT08.S3(NOx)",      # Metal Oxide sensor
  "PT08.S4(NO2)",      # Metal Oxide sensor
  "PT08.S5(O3)",       # Metal Oxide sensor
  "T",                 # Temperature (°C)
  "RH",                # Relative Humidity (%)
  "AH"                 # Absolute Humidity
]].copy()

df_y_pre: pd.DataFrame = df_air_quality_encoded[['C6H6(GT)']].copy()

print("2.4 Replacing nan by NA")
df_x_pre.replace("nan", pd.NA, inplace=True)
df_y_pre.replace("nan", pd.NA, inplace=True)

print("2.5 Keeping only rows where y is not NAN")
# Keep only rows where target exists
mask = df_y_pre["C6H6(GT)"].notna()
df_x_pre = df_x_pre.loc[mask].copy()
df_y_pre = df_y_pre.loc[mask].copy()

print("3. Creating SimpleImputer, StandardScaler and RidgeCV "
      "and LassoCV objects")
ridge_imputer: SimpleImputer = SimpleImputer(strategy="median")
ridge_std_sclr: StandardScaler = StandardScaler()
ridge_cv: RidgeCV = RidgeCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1,
                                    1, 10, 100, 1000, 10000], cv=5)
lasso_imputer: SimpleImputer = SimpleImputer(strategy="median")
lasso_std_sclr: StandardScaler = StandardScaler()
lasso_cv: LassoCV = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1,
                                    1, 10, 100, 1000, 10000], cv=5)

print("4. Creating pipelines")
print("4.1 Creating pipeline for Ridge")
pipeline_ridge: Pipeline = Pipeline([
    ("imputer", ridge_imputer),
    ("std_scaler", ridge_std_sclr),
    ("rigde_cv", ridge_cv)
])
print("4.2 Creating pipeline for Lasso")
pipeline_lasso: Pipeline = Pipeline([
    ("imputer", lasso_imputer),
    ("std_scaler", lasso_std_sclr),
    ("lasso_cv", lasso_cv)
])

print("5. Getting training and testing sets")
x_train_pre: pd.DataFrame
x_test_pre: pd.DataFrame
y_train_pre: pd.DataFrame
y_test_pre: pd.DataFrame

x_train_pre, x_test_pre, y_train_pre, y_test_pre = train_test_split(
    df_x_pre, df_y_pre, train_size=0.6, random_state=42)

print("6. Ridge")
print("6.1 Training Ridge model")
pipeline_ridge.fit(x_train_pre, y_train_pre.to_numpy().ravel())
print("6.2 Getting accuracy score")
score = pipeline_ridge.score(x_test_pre, y_test_pre)
print("Accuracy score = {0}".format(score))
print("6.3 Getting best alpha")
best_alpha = pipeline_ridge.named_steps["rigde_cv"].alpha_
print("Best alpha:", best_alpha)
print("6.4 Predicting x_test_pre")
y_pred_ridge: np.ndarray = pipeline_ridge.predict(x_test_pre)
print("6.5 Model stats")
regression_results(y_test_pre, y_pred_ridge, "Ridge")

print("7. Lasso")
print("7.1 Training the model")
pipeline_lasso.fit(x_train_pre, y_train_pre.to_numpy().ravel())
print("7.2 Getting accuracy score")
score = pipeline_lasso.score(x_test_pre, y_test_pre)
print("Accuracy score = {0}".format(score))
print("7.3 Getting best alpha")
best_alpha = pipeline_lasso.named_steps["lasso_cv"].alpha_
print("Best alpha:", best_alpha)
print("7.4 Predicting x_test_pre")
y_pred_lasso: np.ndarray = pipeline_lasso.predict(x_test_pre)
print("7.5 Getting model stats")
regression_results(y_test_pre, y_pred_lasso, "Lasso")

print("8. Visual Representations")
y_test_plot = y_test_pre.reset_index(drop=True)
y_pred_ridge_series: pd.Series = pd.Series(y_pred_ridge.ravel())
y_pred_lasso_series: pd.Series = pd.Series(y_pred_lasso.ravel())
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
create_plot_scatter_actual_vs_predicted(axes, 0, 0, y_test_plot,
                                        y_pred_ridge_series,
                                        "Ridge", "Ridge Regression")
create_plot_line_actual_vs_predicted(axes, 1, 0, y_test_plot,
                                     y_pred_ridge_series,
                                     "Ridge", "Ridge vs Actual", "green")

create_plot_scatter_actual_vs_predicted(axes, 0, 1, y_test_plot,
                                        y_pred_lasso_series,
                                        "Lasso", "Lasso Regression")
create_plot_line_actual_vs_predicted(axes, 1, 1, y_test_plot,
                                     y_pred_lasso_series,
                                     "Lasso", "Lasso vs Actual", "blue")
plt.show()
