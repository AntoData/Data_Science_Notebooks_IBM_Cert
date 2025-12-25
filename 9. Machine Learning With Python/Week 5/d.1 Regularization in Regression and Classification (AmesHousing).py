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
SOURCE: https://jse.amstat.org/v19n3/decock/AmesHousing.xls
Data set contains information from the Ames Assessor’s Office used in 
computing assessed values for individual residential properties sold in 
Ames, IA from 2006 to 2010.

More details on data: 
https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

With this dataset we will create a regression problem using all features
to predict the price of a house so:

Variable x: Independent variables
All columns except SalePrice

Variable y: Dependent variable
Column SalePrice

We will split the dataset in training and testing sets

We will first create 3 models:
- Linear Regression
- Linear Ridge Regression
- Linear Lasso Regression

We will display their:
- Explained variance
- R^2 score
- Mean absolute error (MAE)
- Mean Square Error (MSE)
- Root Mean Square Error (RMSE)

so we can compare performances

We will display the coefficients of Ridge and Lasso model compared to
Linear model's coefficients
Later we will apply cross validation to these models to get R^2 score
and compare again

Finally, we will filter feature using Lasso. Basically, we will remove
any feature whose coefficient is 0 in the Lasso model and repeat 
these steps and see how models performance varies
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
df_ames_housing: pd.DataFrame = pd.read_excel('AmesHousing.xls')
print("1.1 Let's check its content")
print(df_ames_housing.head())

print("2. Data preprocessing")
print("2.1 Removing columns that are not needed for this problem")
drop_columns: [str] = ["Order", "PID"]
df_ames_housing.drop(columns=drop_columns, inplace=True)
print("2.2 Turning all categorical features into numeric ones using"
      " get_dummies")
df_ames_housing_encoded: pd.DataFrame = pd.get_dummies(df_ames_housing,
                                                       drop_first=True)
print("2.3 Recheck now columns, types and data")
print(df_ames_housing_encoded.columns)
print(df_ames_housing_encoded.head())
print(df_ames_housing_encoded.dtypes)


print("2.4 Getting variable x and y")
df_x_pre: pd.DataFrame = df_ames_housing_encoded.drop(columns='SalePrice')
df_y_pre: pd.DataFrame = df_ames_housing_encoded[['SalePrice']]

print("2.5 Getting training and testing sets")
x_train_pre, x_test_pre, y_train_pre, y_test_pre \
    = train_test_split(df_x_pre, df_y_pre, random_state=42)

print("2.6 Replacing NA values by median")
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

print("2.7 Getting p-values and correlation coefficients for each feature")
for col in x_train_pre.columns:
    pearson_coef, p_value = scipy.stats.pearsonr(
        x_train_pre[col].to_numpy().ravel(),
        y_train_pre[['SalePrice']].to_numpy().ravel())
    print(f"{col}: Pearson Coefficient = {pearson_coef: .4f}, "
          f"p-value = {p_value: .4e}")

print("2.8 Getting only numeric features of train set to "
      "apply Standard Scaler later")
x_train_pre_numeric_columns: pd.Index = \
    x_train_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_train_pre_numeric_columns)
print("2.9 Getting non-numeric features, in this case all boolean features")
x_train_pre_non_numeric_cols: pd.Index = x_train_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_train_pre_non_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_non_numeric_cols]
df_x_train_pre_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_numeric_columns]
print("")

print("2.10 Applying standard scaler to numeric features in train set")
std_sclr: StandardScaler = StandardScaler()
std_sclr.fit(df_x_train_pre_numeric)
x_train_numeric_scaled: np.ndarray = std_sclr.transform(df_x_train_pre_numeric)

print("2.11 Binding non numeric features with scaled numeric features again")
df_x_train_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_train_numeric_scaled,
    columns=x_train_pre_numeric_columns,
    index=x_train_pre[x_train_pre_numeric_columns].index
)

df_x_train_scaled: pd.DataFrame = pd.concat([df_x_train_numeric_scaled,
                                             df_x_train_pre_non_numeric],
                                            axis=1)
print("2.12 Getting only numeric features of test set to "
      "apply Standard Scaler later")
x_test_pre_numeric_columns: pd.Index = \
    x_test_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_test_pre_numeric_columns)
print("2.13 Getting non-numeric features, in this case all boolean features")
x_test_pre_non_numeric_cols: pd.Index = x_test_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_test_pre_non_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_non_numeric_cols]
df_x_test_pre_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_numeric_columns]

print("2.14 Applying Standard scale already trained for train set")
x_test_numeric_scaled: np.ndarray = std_sclr.transform(df_x_test_pre_numeric)

print("2.15 Binding non numeric features with scaled numeric features again")
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
lm_lasso: Lasso = Lasso(alpha=0.01, max_iter=10000)
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
print("5.1 We need to rescaled the whole dataset")
print("5.1.1 Getting only numeric features to apply Standard Scaler later")
df_ames_housing_numeric: pd.Index = \
    df_ames_housing_encoded.select_dtypes(include=['int64', 'float64']).drop(
        columns=["SalePrice"]).columns
print(df_ames_housing_numeric)
print("5.1.2 Getting non-numeric features, in this case all boolean features")
df_ames_housing_non_numeric: pd.Index = df_ames_housing_encoded. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_ames_housing_non_numeric: pd.DataFrame = df_ames_housing_encoded[
    df_ames_housing_non_numeric]
print("")

print("5.1.2 Applying standard scaler to numeric features")
std_sclr: StandardScaler = StandardScaler()
std_sclr.fit(df_ames_housing_encoded[df_ames_housing_numeric])
numeric_scaled_array: np.ndarray = std_sclr.transform(
    df_ames_housing_encoded[df_ames_housing_numeric])

print("5.1.3 Binding non numeric features with scaled numeric features again")
df_numeric_scaled: pd.DataFrame = pd.DataFrame(
    numeric_scaled_array,
    columns=df_ames_housing_numeric,
    index=df_ames_housing_encoded[df_ames_housing_numeric].index
)
df_ames_housing_preprocessed = pd.concat([df_numeric_scaled,
                                          df_ames_housing_non_numeric], axis=1)
print(df_ames_housing_preprocessed)

print("5.1.4 Getting variables x and y")
df_x: pd.DataFrame = df_ames_housing_preprocessed.copy()
df_y: pd.Series = df_ames_housing_encoded['SalePrice'].copy()

x_imputer: SimpleImputer = SimpleImputer(strategy="median")

x_imputer.fit(df_x)

df_x: pd.DataFrame = pd.DataFrame(
    x_imputer.transform(df_x),
    columns=df_x.columns,
    index=df_x.index,
)

y_imputer: SimpleImputer = SimpleImputer(strategy="median")
y_imputer.fit(df_y.to_frame())

df_y: pd.DataFrame = pd.DataFrame(
    y_imputer.transform(df_y.to_frame()),
    columns=["SalePrice"],
    index=df_y.index,
)

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

lasso_all: Lasso = Lasso(alpha=0.0015)
scores_lasso: np.ndarray = cross_val_score(lasso_all, df_x, df_y, cv=5)
print("Lasso Scores")
print(scores_lasso)
print(np.mean(scores_lasso))

print("6. Post-processing")
print("6.1 Getting model's coefficients")
linear_coefficients: np.ndarray = lm.coef_
ridge_coefficients: np.ndarray = lm_rid.coef_
lasso_coefficients: np.ndarray = lm_lasso.coef_

print("6.2 Comparing Lasso and Ridge coefficients to Linear")
models_coefficients_: dict = {"Linear": linear_coefficients,
                              "Ridge": ridge_coefficients,
                              "Lasso": lasso_coefficients}
colours_: [str] = ["red", "blue", "green"]

create_plot_compare_coefficients(models_coefficients_, colours_)

print("6.3 Applying Lasso reduction, removing every feature that is 0 "
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

print("6.4 Dividing the newly filtered data in training and testing sets")
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

print("7. Training the models with the filtered data")
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
lm_lasso_sel: Lasso = Lasso(alpha=0.0015)
lm_lasso_sel.fit(x_train_sel, y_train_sel)
y_pred_lasso_sel: np.ndarray = lm_lasso_sel.predict(x_test_sel)
regression_results(y_test_sel, y_pred_lasso_sel, "Lasso")

print("8. Plotting predictions vs actual data (after Lasso optimisation)")
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

print("9. Cross validation: Getting R^2 score after Lasso optimisation")
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

lasso_all_sel: Lasso = Lasso(alpha=0.0015)
scores_lasso_sel: np.ndarray = cross_val_score(lasso_all_sel,
                                               df_filtered_data, df_y, cv=5)
print("Lasso Scores")
print(scores_lasso_sel)
print(np.mean(scores_lasso_sel))

print("10. Plotting Lasso and Ridge coefficients compared to Linear")
linear_coefficients_sel: np.ndarray = lm_sel.coef_
ridge_coefficients_sel: np.ndarray = lm_rid_sel.coef_
lasso_coefficients_sel: np.ndarray = lm_lasso_sel.coef_

models_coefficients_sel: dict = {"Linear": linear_coefficients_sel,
                                 "Ridge": ridge_coefficients_sel,
                                 "Lasso": lasso_coefficients_sel}
colours_: [str] = ["red", "blue", "green"]

create_plot_compare_coefficients(models_coefficients_sel, colours_)
