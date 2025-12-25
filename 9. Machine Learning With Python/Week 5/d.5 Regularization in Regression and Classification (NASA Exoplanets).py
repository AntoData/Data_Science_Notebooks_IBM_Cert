import pandas as pd
import numpy as np
import scipy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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

    # The "Messy" Metadata & Errors (Adding Noise)
    'sy_pnum',  # Number of planets
    'sy_snum',  # Number of stars
    'glon', 'glat',  # Galactic coordinates
    'ra', 'dec',  # Sky coordinates
    'st_age',  # Stellar age (notoriously poorly measured)

    # Adding uncorrelated column Orbital Period [days] to add noise
    'pl_orbper'
]
print("2.5.4 Building variables x and y")
df_x_pre: pd.DataFrame = df_exoplanets_nasa_encoded[x_cols]
df_y_pre: pd.DataFrame = df_exoplanets_nasa_encoded[y_col]

print("2.6 Getting training and testing sets")
x_train_pre, x_test_pre, y_train_pre, y_test_pre \
    = train_test_split(df_x_pre, df_y_pre, random_state=42)

print("2.7 Replacing NA values by median")
x_imputer: SimpleImputer = SimpleImputer(strategy="median")

x_imputer.fit(x_train_pre)

x_train_pre: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(x_train_pre),
    columns=x_train_pre.columns,
    index=x_train_pre.index,
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
print("2.13 Getting only numeric features of test set to "
      "apply Standard Scaler later")
x_test_pre_numeric_columns: pd.Index = \
    x_test_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_test_pre_numeric_columns)
print("2.14 Getting non-numeric features, in this case all boolean features")
x_test_pre_non_numeric_cols: pd.Index = x_test_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_test_pre_non_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_non_numeric_cols]
df_x_test_pre_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_numeric_columns]

print("2.15 Applying Standard scale already trained for train set")
x_test_numeric_scaled: np.ndarray = std_sclr.transform(df_x_test_pre_numeric)

print("2.16 Binding non numeric features with scaled numeric features again")
df_x_test_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_test_numeric_scaled,
    columns=x_test_pre_numeric_columns,
    index=x_test_pre[x_test_pre_numeric_columns].index
)

df_x_test_scaled: pd.DataFrame = pd.concat([df_x_test_numeric_scaled,
                                            df_x_test_pre_non_numeric],
                                           axis=1)

x_train = df_x_train_scaled
x_test = df_x_test_scaled
y_train = y_train_pre
y_test = y_test_pre

print("3. Training our models")
print("Linear Regression")
lm: LinearRegression = LinearRegression()
lm.fit(x_train, y_train)
y_pred_linear: np.ndarray = lm.predict(x_test)
regression_results(y_test, y_pred_linear, "Linear")

print("Ridge Regression")
lm_rid: Ridge = Ridge(alpha=10)
lm_rid.fit(x_train, y_train)
y_pred_ridge: np.ndarray = lm_rid.predict(x_test)
regression_results(y_test, y_pred_ridge, "Ridge")

print("Lasso Regression")
lm_lasso: Lasso = Lasso(alpha=0.02, max_iter=10000)
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

print("5. Cross validation: Getting R^2 scores")
print("5.1 Building variables x and y from x_cols / y_col")
df_x_pre: pd.DataFrame = df_exoplanets_nasa_encoded.loc[:, x_cols].copy()
df_y_pre: pd.DataFrame = df_exoplanets_nasa_encoded.loc[:, y_col].copy()

print("5.2 Getting numeric vs non-numeric columns within x_cols")
x_numeric_cols: [str] = \
    df_x_pre.select_dtypes(include=["int64", "float64"]).columns
x_non_numeric_cols: [str] = df_x_pre.columns.difference(x_numeric_cols)

df_x_num: pd.DataFrame = df_x_pre.loc[:, x_numeric_cols].copy()
df_x_non: pd.DataFrame = df_x_pre.loc[:, x_non_numeric_cols].copy()

print("5.3 Applying StandardScaler to numeric X only")
std_sclr: StandardScaler = StandardScaler()
numeric_scaled_array: np.ndarray = std_sclr.fit_transform(df_x_num)

df_x_num_scaled: pd.DataFrame = pd.DataFrame(
    numeric_scaled_array,
    columns=df_x_num.columns,
    index=df_x_num.index,
)

print("5.4 Recombining scaled numeric with non-numeric X")
df_x: pd.DataFrame = pd.concat([df_x_num_scaled, df_x_non], axis=1)

# IMPORTANT: ensure column order stays consistent (optional but recommended)
df_x = df_x.loc[:, list(df_x_num_scaled.columns) + list(df_x_non.columns)]

print("5.5 Imputing variable")
# Recompute which columns are numeric after recombination / dropping
x_numeric_cols2: [str] = df_x.select_dtypes(
    include=["int64", "float64"]).columns
x_non_numeric_cols2: [str] = df_x.columns.difference(x_numeric_cols2)

print("5.6 Replacing NA with median for numeric columns")
# Impute numeric part with median
x_num_imputer: SimpleImputer = SimpleImputer(strategy="median")
df_x_num_imputed: pd.DataFrame = pd.DataFrame(
    x_num_imputer.fit_transform(df_x.loc[:, x_numeric_cols2]),
    columns=x_numeric_cols2,
    index=df_x.index,
)

print("5.7 Replacing NA with most frequent or bool columns")
# Impute non-numeric part (bool) with most_frequent
if len(x_non_numeric_cols2) > 0:
    x_non_imputer: SimpleImputer = SimpleImputer(strategy="most_frequent")
    df_x_non_imputed: pd.DataFrame = pd.DataFrame(
        x_non_imputer.fit_transform(df_x.loc[:, x_non_numeric_cols2]),
        columns=x_non_numeric_cols2,
        index=df_x.index,
    )
else:
    df_x_non_imputed: pd.DataFrame = pd.DataFrame(index=df_x.index)

print("5.8 Binding columns for variable x again")
df_x: pd.DataFrame = pd.concat([df_x_num_imputed, df_x_non_imputed], axis=1)
df_x: pd.DataFrame = \
    df_x.loc[:, list(x_numeric_cols2) + list(x_non_numeric_cols2)]

print("5.9 Replacing NA for median for variable y")
y_imputer: SimpleImputer = SimpleImputer(strategy="median")
df_y: pd.DataFrame = pd.DataFrame(
    y_imputer.fit_transform(df_y_pre),
    columns=y_col,
    index=df_y_pre.index,
)

print("Done. Shapes:")
print("X:", df_x.shape)
print("y:", df_y.shape)

print("6. Training the models again and using cross validation for get R2")
lin_model_all: LinearRegression = LinearRegression()
scores_lin: np.ndarray = cross_val_score(lin_model_all, df_x, df_y, cv=5)
print("Linear Scores")
print(scores_lin)
print(np.mean(scores_lin))

ridge_all: Ridge = Ridge(alpha=10)
scores_ridge: np.ndarray = cross_val_score(ridge_all, df_x, df_y, cv=5)
print("Ridge Scores")
print(scores_ridge)
print(np.mean(scores_ridge))

lasso_all: Lasso = Lasso(alpha=0.02)
scores_lasso: np.ndarray = cross_val_score(lasso_all, df_x, df_y, cv=5)
print("Lasso Scores")
print(scores_lasso)
print(np.mean(scores_lasso))

print("7. Post-processing")
print("7.1 Getting model's coefficients")
linear_coefficients: np.ndarray = lm.coef_
ridge_coefficients: np.ndarray = lm_rid.coef_
lasso_coefficients: np.ndarray = lm_lasso.coef_

print("7.2 Comparing Lasso and Ridge coefficients to Linear")
models_coefficients_: dict = {"Linear": linear_coefficients,
                              "Ridge": ridge_coefficients,
                              "Lasso": lasso_coefficients}
colours_: [str] = ["red", "blue", "green"]

create_plot_compare_coefficients(models_coefficients_, colours_)

print("7.3 Applying Lasso reduction, removing every feature that is 0 "
      "in Lasso's coefficients in the other models")
df_linear_coef: pd.DataFrame = pd.DataFrame(
    {"coefficients": linear_coefficients.ravel()})
df_ridge_coef: pd.DataFrame = pd.DataFrame(
    {"coefficients": ridge_coefficients.ravel()})
df_lasso_coef: pd.DataFrame = pd.DataFrame(
    {"coefficients": lasso_coefficients.ravel()})

# To filter by Lasso, a typical strategy is to only keep the features
# whose coefficients are not 0.
# In this case, we keep the indexes of columns whose coefficient is not
# 0
ind_selected_features: [int] = \
    df_lasso_coef[df_lasso_coef["coefficients"] != 0].index
# We use the indexes to keep only those columns using iloc
df_filtered_data: pd.DataFrame = df_x.iloc[:, ind_selected_features]

print("7.4 Dividing the newly filtered data in training and testing sets")
x_train_sel, x_test_sel, y_train_sel, y_test_sel = train_test_split(
    df_filtered_data, df_y, test_size=0.6, random_state=42)

x_imputer: SimpleImputer = SimpleImputer(strategy="median")

x_imputer.fit(x_train_sel)

x_train_sel: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(x_train_sel),
    columns=x_train_sel.columns,
    index=x_train_sel.index,
)

x_test_sel: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(x_test_sel),
    columns=x_test_sel.columns,
    index=x_test_sel.index,
)

y_imputer: SimpleImputer = SimpleImputer(strategy="median")
y_imputer.fit(y_train_sel)

y_train_sel: pd.DataFrame = pd.DataFrame(
    y_imputer.transform(y_train_sel),
    columns=y_train_sel.columns,
    index=y_train_sel.index,
)

y_test_sel: pd.DataFrame = pd.DataFrame(
    y_imputer.transform(y_test_sel),
    columns=y_test_sel.columns,
    index=y_test_sel.index,
)

print("8. Training the models with the filtered data")
print("Linear Regression (Filtered coefficients)")
lm_sel: LinearRegression = LinearRegression()
lm_sel.fit(x_train_sel, y_train_sel)
y_pred_linear_sel: np.ndarray = lm_sel.predict(x_test_sel)
regression_results(y_test_sel, y_pred_linear_sel, "Linear (Filtered "
                                                  "coefficients)")

print("Ridge Regression (Filtered coefficients)")
lm_rid_sel: Ridge = Ridge(alpha=10)
lm_rid_sel.fit(x_train_sel, y_train_sel)
y_pred_ridge_sel: np.ndarray = lm_rid_sel.predict(x_test_sel)
regression_results(y_test_sel, y_pred_ridge_sel, "Ridge (Filtered "
                                                 "coefficients)")

print("Lasso Regression (Filtered coefficients)")
lm_lasso_sel: Lasso = Lasso(alpha=0.02)
lm_lasso_sel.fit(x_train_sel, y_train_sel)
y_pred_lasso_sel: np.ndarray = lm_lasso_sel.predict(x_test_sel)
regression_results(y_test_sel, y_pred_lasso_sel, "Lasso")

print("9. Plotting predictions vs actual data (after Lasso optimisation)")
y_test_plot_sel = y_test_sel.reset_index(drop=True)
y_pred_linear_sel_series = pd.Series(y_pred_linear_sel.ravel())
y_pred_ridge_sel_series = pd.Series(y_pred_ridge_sel.ravel())
y_pred_lasso_sel_series = pd.Series(y_pred_lasso_sel.ravel())
_, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
create_plot_scatter_actual_vs_predicted(axes, 0, 0, y_test_plot_sel,
                                        y_pred_linear_sel_series,
                                        "Linear", "Linear Regression")
create_plot_scatter_actual_vs_predicted(axes, 0, 1, y_test_plot_sel,
                                        y_pred_ridge_sel_series,
                                        "Ridge", "Ridge Regression")
create_plot_scatter_actual_vs_predicted(axes, 0, 2, y_test_plot_sel,
                                        y_pred_lasso_sel_series,
                                        "Lasso", "Lasso Regression")
create_plot_line_actual_vs_predicted(axes, 1, 0, y_test_plot_sel,
                                     y_pred_linear_sel_series,
                                     "Linear", "Linear vs Actual", "red")
create_plot_line_actual_vs_predicted(axes, 1, 1, y_test_plot_sel,
                                     y_pred_ridge_sel_series,
                                     "Ridge", "Ridge vs Actual", "green")
create_plot_line_actual_vs_predicted(axes, 1, 2, y_test_plot_sel,
                                     y_pred_lasso_sel_series,
                                     "Lasso", "Lasso vs Actual", "red")
plt.show()

print("10. Cross validation: Getting R^2 score after Lasso optimisation")
lin_model_all_sel: LinearRegression = LinearRegression()
scores_lin_sel: np.ndarray = cross_val_score(
    lin_model_all_sel, df_filtered_data, df_y, cv=5)
print("Linear Scores")
print(scores_lin_sel)
print(np.mean(scores_lin_sel))

ridge_all_sel: Ridge = Ridge(alpha=10)
scores_ridge_sel: np.ndarray = cross_val_score(ridge_all_sel,
                                               df_filtered_data, df_y, cv=5)
print("Ridge Scores")
print(scores_ridge_sel)
print(np.mean(scores_ridge_sel))

lasso_all_sel: Lasso = Lasso(alpha=0.022)
scores_lasso_sel: np.ndarray = cross_val_score(lasso_all_sel,
                                               df_filtered_data, df_y, cv=5)
print("Lasso Scores")
print(scores_lasso_sel)
print(np.mean(scores_lasso_sel))

print("11. Plotting Lasso and Ridge coefficients compared to Linear")
linear_coefficients_sel: np.ndarray = lm_sel.coef_
ridge_coefficients_sel: np.ndarray = lm_rid_sel.coef_
lasso_coefficients_sel: np.ndarray = lm_lasso_sel.coef_

models_coefficients_sel: dict = {"Linear": linear_coefficients_sel,
                                 "Ridge": ridge_coefficients_sel,
                                 "Lasso": lasso_coefficients_sel}
colours_: [str] = ["red", "blue", "green"]

create_plot_compare_coefficients(models_coefficients_sel, colours_)
