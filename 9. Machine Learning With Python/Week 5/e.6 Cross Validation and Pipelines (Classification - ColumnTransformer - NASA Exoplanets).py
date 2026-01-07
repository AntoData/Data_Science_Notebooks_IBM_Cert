import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, \
    explained_variance_score, mean_absolute_error

"""
SOURCE: exoplanetarchive.ipac.caltech.edu
Dataset: NASA Planetary Systems (Confirmed Planets)
The dataset contains astrophysical parameters for all confirmed 
exoplanets discovered and archived by NASA as of late 2025. 
It includes orbital characteristics, planetary physical measurements, 
and host star properties.

More details on data: 
exoplanetarchive.ipac.caltech.edudocs/API_PS_columns.html

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
- Handling multicollinearity by including redundant stellar properties.
- Standardizing numeric features using StandardScaler.

We will split the dataset into training and testing sets and create 3 
models:
- Linear Regression (Ordinary Least Squares)
- Linear Ridge Regression (L2 Regularization)
- Linear Lasso Regression (L1 Regularization)

We will display their evaluation metrics:
- Explained variance
- R^2 score
- Mean Absolute Error (MAE)
- Mean Square Error (MSE)
- Root Mean Square Error (RMSE)

We will compare the models to demonstrate how Lasso handles 
high-dimensional noise and multicollinearity to maintain a high R^2
 (~0.77) while standard linear models may degrade or overfit.

We will display the coefficients of Ridge and Lasso models compared to 
Linear model's coefficients to identify the "Primary Physical Laws" 
discovered by the models.

Later, we will apply cross-validation to these models to assess the 
robustness of the R^2 scores.

Finally, we will perform feature selection using Lasso by removing 
features with zero coefficients, demonstrating how Lasso simplifies the
 scientific model without losing predictive power.
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


def create_plot_compare_coefficients(models_coefficients: dict,
                                     colours: [str]) -> None:
    """
    Creates a plot with two subplots, one in the top and one at the
    bottom. First one displays coefficients in a bar plot. Second one,
    displays coefficient residuals compared to regular simple Linear
    model

    :param models_coefficients: Dictionary where key is a key with the
    model name and value is the list of coefficients of that model. Note
    we need to have one called Linear and another called Lasso
    :type models_coefficients: dict
    :param colours: Array with colour to represent each model in the
    plot (so it needs to have the same length that models_coefficients)
    :type colours: [str]
    :return: None
    """

    print("create_plot_compare_coefficients: ç"
          "Normalising coefficients to 1D arrays")
    coeffs_1d: dict[str, np.ndarray] = {}
    for label, coefficients in models_coefficients.items():
        coef_1d = np.asarray(coefficients).ravel()
        coeffs_1d[label] = coef_1d
        print(f" - {label} coefficients shape: "
              f"{np.asarray(coefficients).shape} -> {coef_1d.shape}")

    if "Linear" not in coeffs_1d:
        raise ValueError("create_plot_compare_coefficients: "
                         "models_coefficients must include a 'Linear' key")

    print("create_plot_compare_coefficients: "
          "Building x-axis based on number of FEATURES")
    n_features: int = coeffs_1d["Linear"].shape[0]
    x_axis: np.ndarray = np.arange(n_features)

    # Make readable ticks for both small and large n_features
    step: int = 10 if n_features >= 100 else max(1, n_features // 10)
    x_labels: np.ndarray = np.arange(0, n_features, step)

    # Safety check: all coefficient vectors must match feature length
    for label, coef in coeffs_1d.items():
        if coef.shape[0] != n_features:
            raise ValueError(
                f"create_plot_compare_coefficients: '{label}' has "
                f"{coef.shape[0]} coefficients "
                f"but Linear has {n_features}. ("
                f"Your models were not trained on identical X columns.)"
            )

    n_models: int = len(coeffs_1d)
    bar_width: float = 0.8 / n_models

    fig_, axes_ = plt.subplots(2, 1, figsize=(18, 10), sharey=True)

    print("create_plot_compare_coefficients: Plotting coefficients")
    i: int = 0
    for label, coef in coeffs_1d.items():
        offset: float = (i - n_models / 2) * bar_width + bar_width / 2
        axes_[0].bar(x_axis - offset, coef, width=bar_width,
                     label=label, color=colours[i])
        i += 1

    axes_[0].set_title("Comparison of Model Coefficients")
    axes_[0].set_xlabel("Feature Index")
    axes_[0].set_ylabel("Coefficient Value")
    axes_[0].set_xticks(x_labels)
    axes_[0].legend()

    print("create_plot_compare_coefficients: Plotting residuals vs Linear")
    linear_coef = coeffs_1d["Linear"]
    j: int = 0
    for label, coef in coeffs_1d.items():
        if label == "Linear":
            continue
        residual = linear_coef - coef
        if label == "Lasso":
            axes_[1].plot(x_axis, residual, label=label, color=colours[j])
        else:
            axes_[1].bar(x_axis, residual, width=bar_width, label=label,
                         color=colours[j])
        j += 1

    axes_[1].set_xlabel("Feature Index")
    axes_[1].set_ylabel("Coefficient Value")
    axes_[1].set_title("Comparison of Model Coefficient Residuals")
    axes_[1].set_xticks(x_labels)
    axes_[1].legend()

    plt.show()


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
                                        (df_exoplanets_nasa['pl_rade'] > 0)]

print("2.3 Recheck now columns, types and data")
df_exoplanets_nasa.dropna(axis=1, how="all", inplace=True)
print(df_exoplanets_nasa.columns)
print(df_exoplanets_nasa.head())
print(df_exoplanets_nasa.dtypes)

print("2.4 Getting variable x and y")
print("2.4.1 Variable y -> log10 of pl_bmassj")
# We predict log-mass because planetary masses vary from 0.0001 to 30+
# Jupiter masses.
df_exoplanets_nasa['log_y'] = np.log10(
    df_exoplanets_nasa['pl_bmassj'])

print("2.4.2 We need to apply log10 on log_pl_orbper and log_pl_rade "
      "for variable x")
# Key physical features also follow power laws (e.g.,
# period-mass relationships).
df_exoplanets_nasa['log_pl_orbper'] = np.log10(
    df_exoplanets_nasa['pl_orbper'])
df_exoplanets_nasa['log_pl_rade'] = np.log10(
    df_exoplanets_nasa['pl_rade'])

print("2.4.3 Getting columns for x and y")
y_col: [str] = ['log_y']
x_cols: [str] = [
    # The "Good" Physical Predictors
    'log_pl_orbper', 'log_pl_rade', 'st_teff', 'st_mass', 'st_lum', 'sy_dist',
    # Including non-numeric features so it uses the non-numeric pipeline
    "pl_name"]

print("2.4.4 Building variables x and y")
df_x_pre: pd.DataFrame = df_exoplanets_nasa[x_cols]
df_y_pre: pd.DataFrame = df_exoplanets_nasa[y_col]

print("2.5 Getting only numeric features of train set to "
      "apply Standard Scaler later")
x_pre_numeric_columns: pd.Index = \
    df_x_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_pre_numeric_columns)
print("2.6 Getting non-numeric features, in this case all boolean features")
x_pre_non_numeric_cols: pd.Index = df_x_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
print(x_pre_non_numeric_cols)

print("2.7.1 Creating numerical transformer pipeline for Ridge")
numerical_transformer_ridge: Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
print("2.8.1 Creating non-numerical transformer pipeline for Ridge")
categorical_transformer_ridge: Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
print("2.9.1 Joining those two to create a preprocessor pipeline for Ridge")
preprocessor_ridge: ColumnTransformer = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_ridge, x_pre_numeric_columns),
        ('cat', categorical_transformer_ridge, x_pre_numeric_columns)
    ])
print("2.10.1 Creating RidgeCV object")
ridge_cv: RidgeCV = RidgeCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1,
                                    1, 10, 100, 1000, 10000], cv=5)
print("2.7.2 Creating numerical transformer pipeline for Lasso")
numerical_transformer_lasso: Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
print("2.8.2 Creating non-numerical transformer pipeline for Lasso")
categorical_transformer_lasso: Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
print("2.9.2 Joining those two to create a preprocessor pipeline for Lasso")
preprocessor_lasso: ColumnTransformer = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_lasso, x_pre_numeric_columns),
        ('cat', categorical_transformer_lasso, x_pre_numeric_columns)
    ])
print("2.10.2 Creating LassoCV object")
lasso_cv: LassoCV = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1,
                                    1, 10, 100, 1000, 10000], cv=5, verbose=2)
print("3. Creating pipelines")
print("3.1 Creating pipeline for Ridge")
pipeline_ridge: Pipeline = Pipeline([
    ("preprocessor", preprocessor_ridge),
    ("PCA", PCA()),
    ("ridge_cv", ridge_cv)
])
print("3.2 Creating pipeline for Lasso")
pipeline_lasso: Pipeline = Pipeline([
    ("preprocessor", preprocessor_lasso),
    ("PCA", PCA()),
    ("lasso_cv", lasso_cv)
])

print("4. Getting training and testing sets")
x_train_pre: pd.DataFrame
x_test_pre: pd.DataFrame
y_train_pre: pd.DataFrame
y_test_pre: pd.DataFrame

x_train_pre, x_test_pre, y_train_pre, y_test_pre = train_test_split(
    df_x_pre, df_y_pre, train_size=0.6, random_state=42)


print("5. Ridge")
print("5.1 Training Ridge model")
pipeline_ridge.fit(x_train_pre, y_train_pre.to_numpy().ravel())
print("5.2 Getting accuracy score")
score: float = pipeline_ridge.score(x_test_pre, y_test_pre)
print("Accuracy score = {0}".format(score))
print("5.3 Getting best alpha")
best_alpha: float = pipeline_ridge.named_steps["ridge_cv"].alpha_
print("Best alpha:", best_alpha)
print("5.4 Predicting x_test_pre")
y_pred_ridge: np.ndarray = pipeline_ridge.predict(x_test_pre)
print("5.5 Model stats")
regression_results(y_test_pre, y_pred_ridge, "Ridge")

print("6. Lasso")
print("6.1 Training the model")
pipeline_lasso.fit(x_train_pre, y_train_pre.to_numpy().ravel())
print("6.2 Getting accuracy score")
score: float = pipeline_lasso.score(x_test_pre, y_test_pre)
print("Accuracy score = {0}".format(score))
print("6.3 Getting best alpha")
best_alpha: float = pipeline_lasso.named_steps["lasso_cv"].alpha_
print("Best alpha:", best_alpha)
print("6.4 Predicting x_test_pre")
y_pred_lasso: np.ndarray = pipeline_lasso.predict(x_test_pre)
print("6.5 Getting model stats")
regression_results(y_test_pre, y_pred_lasso, "Lasso")

print("7. Visual Representations")
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
