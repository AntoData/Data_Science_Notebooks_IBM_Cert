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
SOURCE: exoplanetarchive.ipac.caltech.edu
Dataset: NASA Planetary Systems (Confirmed Planets)
The dataset contains astrophysical parameters for all confirmed 
exoplanets discovered and archived by NASA as of late 2025. 
It includes orbital characteristics, planetary physical measurements, 
and host star properties.

More details on data: 
exoplanetarchive.ipac.caltech.edudocs/API_PS_columns.html

This is an example about how to use RidgeCV and LassoCV with pipelines.

With this dataset we will create a regression problem to predict the 
mass of an exoplanet (log-transformed) using physical and stellar 
predictors:

Variable x: Independent variables
Key features including log-transformed Radius and Orbital Period, 
Stellar Mass, Luminosity, Temperature, and Distance.

Variable y: Dependent variable
Log-transformed Planet Mass (pl_bmassj)

We will perform significant data preprocessing, including:
- Filtering for confirmed planets with complete physical records.
- Applying Log10 transformations to handle power-law distributions and 
outliers.
- Standardizing numeric features using StandardScaler.

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
df_exoplanets_nasa: pd.DataFrame = pd.read_csv(
    'PS_2025.12.24_11.00.04_raw.csv', low_memory=False)
print("1.1 Let's check its content")
print(df_exoplanets_nasa.head())

print("2. Data preprocessing")
print("2.1 Dropping NA values")
# There are enough records to be able to afford this, also there is no
# good option to replace NA values
df_exoplanets_nasa = df_exoplanets_nasa.dropna(
    subset=['pl_bmassj', 'pl_rade', 'pl_orbper'])

# We need to remove outliers/limit cases (e.g., Brown Dwarfs or massive
# stars)
# This keeps the model focused on the "main" planetary relationship
print("2.2 Let's remove outliers, limit cases like brown dwarfs or "
      "massive stars")
print("Filter: Max 20 Jupiter masses")
df_exoplanets_nasa = df_exoplanets_nasa[df_exoplanets_nasa['pl_bmassj'] < 20]
print("Filter: Max 3 Solar Masses")
df_exoplanets_nasa = df_exoplanets_nasa[df_exoplanets_nasa['st_mass'] < 3]
print("Filter: Handle log10 of zero/negative")
# Filter 3: Handle the log10 of zero/negative (if any exist by error)
df_exoplanets_nasa = df_exoplanets_nasa[(df_exoplanets_nasa['pl_bmassj'] > 0) &
                                        (df_exoplanets_nasa['pl_rade'] > 0) &
                                        (df_exoplanets_nasa['pl_orbper'] > 0)]

print("2.3 Turning all categorical features into numeric ones using"
      " get_dummies")
df_exoplanets_nasa_encoded: pd.DataFrame = pd.get_dummies(df_exoplanets_nasa,
                                                          drop_first=True)
print("2.4 Recheck now columns, types and data")
print(df_exoplanets_nasa_encoded.columns)
print(df_exoplanets_nasa_encoded.head())
print(df_exoplanets_nasa_encoded.dtypes)

print("2.5 Getting variable x and y")
print("2.5.1 Variable y -> log10 of pl_bmassj")
# We predict log-mass because planetary masses vary from 0.0001 to 30+
# Jupiter masses.
df_exoplanets_nasa_encoded['log_y'] = np.log10(
    df_exoplanets_nasa_encoded['pl_bmassj'])

print("2.5.2 We need to apply log10 on log_pl_orbper and log_pl_rade "
      "for variable x")
# Key physical features also follow power laws (e.g.,
# period-mass relationships).
df_exoplanets_nasa_encoded['log_pl_orbper'] = np.log10(
    df_exoplanets_nasa_encoded['pl_orbper'])
df_exoplanets_nasa_encoded['log_pl_rade'] = np.log10(
    df_exoplanets_nasa_encoded['pl_rade'])

print("2.5.3 Getting columns for x and y")
y_col = ['log_y']
x_cols = [
    # The "Good" Physical Predictors
    'log_pl_orbper', 'log_pl_rade', 'st_teff', 'st_mass', 'st_lum', 'sy_dist',
]

print("2.5.4 Building variables x and y")
df_x_pre: pd.DataFrame = df_exoplanets_nasa_encoded[x_cols]
df_y_pre: pd.DataFrame = df_exoplanets_nasa_encoded[y_col]

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
