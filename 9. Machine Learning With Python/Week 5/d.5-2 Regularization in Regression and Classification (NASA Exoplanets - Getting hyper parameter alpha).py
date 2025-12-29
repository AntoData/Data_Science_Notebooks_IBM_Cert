import pandas as pd
import numpy as np
import scipy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, \
    explained_variance_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

"""
SOURCE: exoplanetarchive.ipac.caltech.edu
Dataset: NASA Planetary Systems (Confirmed Planets)
The dataset contains astrophysical parameters for all confirmed 
exoplanets discovered and archived by NASA as of late 2025. 
It includes orbital characteristics, planetary physical measurements, 
and host star properties.

More details on data: 
exoplanetarchive.ipac.caltech.edudocs/API_PS_columns.html

This is an example about how to get the best alpha hyperparameter, 
avoiding data snooping. In order to do so, you need to divide
your variables x and y and 3 distinct sets each. A training set 
we use to train the model, a validation set we use to test different
alpha hyperparameters to see which one is the best and finally a 
testing set to test the final performance of the model with the 
best alpha hyperparameter

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

We will split the dataset into training, validation and testing sets 
and create 3 models:
- Linear Regression (Ordinary Least Squares)
- Linear Ridge Regression (L2 Regularization)
- Linear Lasso Regression (L1 Regularization)

We will iterate over different alpha hyperparameters for Ridge and Lasso
We will display their evaluation metrics:
- Explained variance
- R^2 score
- Mean Absolute Error (MAE)
- Mean Square Error (MSE)
- Root Mean Square Error (RMSE)
Pick the alpha hyperparameter with the best R^2 score

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


def train_validate_test_split(var_x: pd.DataFrame, var_y: pd.DataFrame,
                              col_y: [str],
                              train_size: float, val_size: float) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
            pd.DataFrame, pd.DataFrame):
    """
    Given a variable x and a variable y splits those sets into 3
    different sets each:
        - Training set: To train the model
        - Validation set: To test different alpha hyperparameters
        - Test set: To test performance of the final model
    :param var_x: Dataframe that contains the features that compose
    variable x
    :param var_y: Dataframe that contains the feature that composes
    variable x
    :param col_y: Column that forms variable y
    :type col_y: [str]
    :param train_size: Size of the training set
    :param val_size: Size of the validation set (in comparison with the
    test set)
    :return: A tuple with the training, validation and testing set
    for all 2 variables
    """
    x_train_: pd.DataFrame
    x_val_test_: pd.DataFrame
    x_val_: pd.DataFrame
    x_test_: pd.DataFrame
    y_train_: pd.DataFrame
    y_val_test_: pd.DataFrame
    y_val_: pd.DataFrame
    y_test_: pd.DataFrame
    bins_va_test: pd.Categorical

    y_bins: pd.Categorical = \
        pd.qcut(var_y[col_y].to_numpy().ravel(), q=10, duplicates="drop")

    x_train_, x_val_test_, y_train_, y_val_test_, bins_train, bins_val_test = \
        train_test_split(var_x, var_y, y_bins, train_size=train_size,
                         random_state=10, stratify=y_bins)

    x_val_, x_test_, y_val_, y_test_ = train_test_split(x_val_test_,
                                                        y_val_test_,
                                                        train_size=val_size,
                                                        random_state=30,
                                                        stratify=bins_val_test)
    return x_train_, x_val_, x_test_, y_train_, y_val_, y_test_


def best_alpha_based_on_r2(model_type: str, alpha_hyperparameters: [float],
                           x_train_: pd.DataFrame, y_train_: pd.DataFrame,
                           x_val_: pd.DataFrame, y_val_: pd.DataFrame) \
        -> float:
    """
    Returns the best alpha parameter in array of alpha hyperparameters
    for the model selected and those training and validation sets

    :param model_type: Model we will train. Values can be Lasso or Ridge
    :type model_type: str
    :param alpha_hyperparameters: Array with the alpha hyperparameters
    to test for our model
    :type alpha_hyperparameters: [float]
    :param x_train_: Training set for variable x
    :type x_train_: pd.Dataframe
    :param y_train_: Training set for variable y
    :type y_train_: pd.Dataframe
    :param x_val_: Validation set for variable x
    :type x_val_: pd.Dataframe
    :param y_val_: Validation set for variable y
    :type y_val_: pd.Dataframe
    :return: Best alpha hyperparameter for the model
    :rtype: float
    """
    print("Evaluating alpha hyperparameters for model {0}".format(model_type))
    model_alphas_r2: {float: float} = {}
    for alpha_ in alpha_hyperparameters:
        print(f"alpha = {alpha_}")
        if model_type.lower() == "ridge":
            model: Ridge = Ridge(alpha=alpha_)
        elif model_type.lower() == "lasso":
            model: Lasso = Lasso(alpha=alpha_)
        else:
            raise ValueError("model_type valid values are Ridge and Lasso")
        model.fit(x_train_, y_train_)
        y_val_pred_: np.ndarray = model.predict(x_val_)
        model_alphas_r2[alpha_] = r2_score(y_val_, y_val_pred_)
        print("R2 score = {0}".format(model_alphas_r2[alpha_]))
        regression_results(y_val_, y_val_pred_, model_type)
    best_alpha: float = max(model_alphas_r2, key=model_alphas_r2.get)
    return best_alpha


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
pl_name_dummies = [c for c in df_exoplanets_nasa_encoded.columns
                   if c.startswith("pl_name_")]

x_cols = x_cols + pl_name_dummies

print("2.5.4 Building variables x and y")
df_x_pre: pd.DataFrame = df_exoplanets_nasa_encoded[x_cols]
df_y_pre: pd.DataFrame = df_exoplanets_nasa_encoded[y_col]

print("2.6 Getting training, validation and testing sets")
x_train_pre: pd.DataFrame
x_val_pre: pd.DataFrame
x_test_pre: pd.DataFrame
y_train_pre: pd.DataFrame
y_val_pre: pd.DataFrame
y_test_pre: pd.DataFrame

x_train_pre, x_val_pre, x_test_pre, y_train_pre, y_val_pre, y_test_pre = \
    train_validate_test_split(df_x_pre, df_y_pre, y_col,
                              train_size=0.5, val_size=0.5)

print("2.7 Replacing NA values by median")
x_imputer: SimpleImputer = SimpleImputer(strategy="median")

x_imputer.fit(x_train_pre)

x_train_pre: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(x_train_pre),
    columns=x_train_pre.columns,
    index=x_train_pre.index,
)

x_val_pre: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(x_val_pre),
    columns=x_val_pre.columns,
    index=x_val_pre.index,
)

x_test_pre: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(x_test_pre),
    columns=x_test_pre.columns,
    index=x_test_pre.index,
)

y_imputer: SimpleImputer = SimpleImputer(strategy="median")
y_imputer.fit(y_train_pre)

y_train_pre: pd.DataFrame = pd.DataFrame(
    y_imputer.transform(y_train_pre),
    columns=y_train_pre.columns,
    index=y_train_pre.index,
)

y_val_pre: pd.DataFrame = pd.DataFrame(
    y_imputer.transform(y_val_pre),
    columns=y_val_pre.columns,
    index=y_val_pre.index,
)

y_test_pre: pd.DataFrame = pd.DataFrame(
    y_imputer.transform(y_test_pre),
    columns=y_test_pre.columns,
    index=y_test_pre.index,
)

print("2.8 Getting p-values and correlation coefficients for each feature")
for col in x_train_pre.columns:
    pearson_coef, p_value = scipy.stats.pearsonr(
        x_train_pre[col].to_numpy().ravel(),
        y_train_pre[y_col].to_numpy().ravel())
    print(f"{col}: Pearson Coefficient = {pearson_coef: .4f}, "
          f"p-value = {p_value: .4e}")

print("2.9 Getting only numeric features of train set to "
      "apply Standard Scaler later")
x_train_pre_numeric_columns: pd.Index = \
    x_train_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_train_pre_numeric_columns)
print("2.10 Getting non-numeric features, in this case all boolean features")
x_train_pre_non_numeric_cols: pd.Index = x_train_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_train_pre_non_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_non_numeric_cols]
df_x_train_pre_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_numeric_columns]
print("")

print("2.11 Applying standard scaler to numeric features in train set")
std_sclr: StandardScaler = StandardScaler()
std_sclr.fit(df_x_train_pre_numeric)
x_train_numeric_scaled: np.ndarray = std_sclr.transform(df_x_train_pre_numeric)

print("2.12 Binding non numeric features with scaled numeric features again")
df_x_train_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_train_numeric_scaled,
    columns=x_train_pre_numeric_columns,
    index=x_train_pre[x_train_pre_numeric_columns].index
)

df_x_train_scaled: pd.DataFrame = pd.concat([df_x_train_numeric_scaled,
                                             df_x_train_pre_non_numeric],
                                            axis=1)

print("2.13 Getting only numeric features of validation set to "
      "apply Standard Scaler later")
x_val_pre_numeric_columns: pd.Index = \
    x_val_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_val_pre_numeric_columns)
print("2.14 Getting non-numeric features, in this case all boolean features")
x_val_pre_non_numeric_cols: pd.Index = x_val_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_val_pre_non_numeric: pd.DataFrame = x_val_pre[
    x_val_pre_non_numeric_cols]
df_x_val_pre_numeric: pd.DataFrame = x_val_pre[
    x_val_pre_numeric_columns]

print("2.15 Applying Standard scale already trained for train set")
x_val_numeric_scaled: np.ndarray = std_sclr.transform(df_x_val_pre_numeric)

print("2.16 Binding non numeric features with scaled numeric features again")
df_x_val_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_val_numeric_scaled,
    columns=x_val_pre_numeric_columns,
    index=x_val_pre[x_val_pre_numeric_columns].index
)

df_x_val_scaled: pd.DataFrame = pd.concat([df_x_val_numeric_scaled,
                                           df_x_val_pre_non_numeric],
                                          axis=1)

print("2.17 Getting only numeric features of test set to "
      "apply Standard Scaler later")
x_test_pre_numeric_columns: pd.Index = \
    x_test_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_test_pre_numeric_columns)
print("2.18 Getting non-numeric features, in this case all boolean features")
x_test_pre_non_numeric_cols: pd.Index = x_test_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_test_pre_non_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_non_numeric_cols]
df_x_test_pre_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_numeric_columns]

print("2.19 Applying Standard scale already trained for train set")
x_test_numeric_scaled: np.ndarray = std_sclr.transform(df_x_test_pre_numeric)

print("2.20 Binding non numeric features with scaled numeric features again")
df_x_test_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_test_numeric_scaled,
    columns=x_test_pre_numeric_columns,
    index=x_test_pre[x_test_pre_numeric_columns].index
)

df_x_test_scaled: pd.DataFrame = pd.concat([df_x_test_numeric_scaled,
                                            df_x_test_pre_non_numeric],
                                           axis=1)

x_train = df_x_train_scaled
x_val = df_x_val_scaled
x_test = df_x_test_scaled
y_train = y_train_pre
y_val = y_val_pre
y_test = y_test_pre

print("3. Training our models")
print("Linear Regression")
lm: LinearRegression = LinearRegression()
lm.fit(x_train, y_train)
y_pred_linear: np.ndarray = lm.predict(x_test)
regression_results(y_test, y_pred_linear, "Linear")

print("Ridge Regression")
ridge_alphas: [float] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_ridge_alpha: float = best_alpha_based_on_r2(
    "ridge", ridge_alphas, x_train, y_train, x_val, y_val)
print(f"Best alpha for Ridge is: alpha = {best_ridge_alpha}")
print("Training and testing with that value")
lm_rid: Ridge = Ridge(alpha=best_ridge_alpha)
lm_rid.fit(x_train, y_train)
y_pred_ridge: np.ndarray = lm_rid.predict(x_test)
regression_results(y_test, y_pred_ridge, "Ridge")

print("Lasso Regression")
lasso_alphas: [float] = [0.0001, 0.001, 0.01, 0.02, 0.05, 1, 10]
best_lasso_alpha: float = best_alpha_based_on_r2(
    "lasso", lasso_alphas, x_train, y_train, x_val, y_val)
print(f"Best alpha for Lasso is: alpha = {best_lasso_alpha}")
print("Training and testing with that value")
lm_lasso: Lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
lm_lasso.fit(x_train, y_train)
y_pred_lasso: np.ndarray = lm_lasso.predict(x_test)
regression_results(y_test, y_pred_lasso, "Lasso")

print("4. Plotting predictions vs actual values")
y_test_plot = y_test.reset_index(drop=True)
y_pred_linear_series: pd.Series = pd.Series(y_pred_linear.ravel())
y_pred_ridge_series: pd.Series = pd.Series(y_pred_ridge.ravel())
y_pred_lasso_series: pd.Series = pd.Series(y_pred_lasso.ravel())
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
create_plot_scatter_actual_vs_predicted(axes, 0, 0, y_test_plot,
                                        y_pred_linear_series,
                                        "Linear", "Linear Regression")
create_plot_scatter_actual_vs_predicted(axes, 0, 1, y_test_plot,
                                        y_pred_ridge_series,
                                        "Ridge", "Ridge Regression")
create_plot_scatter_actual_vs_predicted(axes, 0, 2, y_test_plot,
                                        y_pred_lasso_series,
                                        "Lasso", "Lasso Regression")
create_plot_line_actual_vs_predicted(axes, 1, 0, y_test_plot,
                                     y_pred_linear_series,
                                     "Linear", "Linear vs Actual", "red")
create_plot_line_actual_vs_predicted(axes, 1, 1, y_test_plot,
                                     y_pred_ridge_series,
                                     "Ridge", "Ridge vs Actual", "green")
create_plot_line_actual_vs_predicted(axes, 1, 2, y_test_plot,
                                     y_pred_lasso_series,
                                     "Lasso", "Lasso vs Actual", "red")
plt.show()
