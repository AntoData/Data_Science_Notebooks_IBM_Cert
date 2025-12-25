import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, \
    explained_variance_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

"""
SOURCE: "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/
prostate.data"

Stamey et al. (1989)
“Prostate specific antigen in the diagnosis and treatment of 
adenocarcinoma of the prostate.”
Journal of Urology.

VARIABLE X
Variables	Description
lcavol	    log cancer volume
lweight	    log prostate weight
age	        patient age
lbph	    log benign prostatic hyperplasia amount
svi	        seminal vesicle invasion (0/1)
lcp	        log capsular penetration
gleason	    Gleason score
pgg45	    % Gleason scores 4 or 5

VARIABLE Y
Variable	Description
lpsa	    log PSA — target variable

We will split the dataset in training and testing sets (already preset 
by dataset)

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
        axes_[0].bar(x_axis - offset, coef, width=bar_width, label=label,
                     color=colours[i])
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
# Source of the dataset
url: str = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
df_prostate: pd.DataFrame = pd.read_csv(url, sep="\t")
df_prostate.drop(inplace=True, columns=["Unnamed: 0"])
print("1.1 Let's check its content")
print(df_prostate.head())

print("2. Data preprocessing")
print("2.1 Turning all categorical features into numeric ones using"
      " get_dummies")
df_prostate_encoded: pd.DataFrame = pd.get_dummies(df_prostate,
                                                   drop_first=True)
print("2.2 Recheck now columns, types and data")
print(df_prostate_encoded.columns)
print(df_prostate_encoded.head())
print(df_prostate_encoded.dtypes)

print("2.3 Dropping NA")
df_prostate_encoded.dropna(inplace=True)

print("2.4 Getting p-values and correlation coefficients for each feature")
df_x_pre: pd.DataFrame = df_prostate_encoded.drop(columns=['lpsa', 'train_T'])
df_y_pre: pd.DataFrame = df_prostate_encoded[['lpsa']]
for col in df_x_pre.columns:
    pearson_coef, p_value = scipy.stats.pearsonr(df_x_pre[col],
                                                 df_y_pre['lpsa'])
    print(f"{col}: Pearson Coefficient ={pearson_coef: .4f}, "
          f"p-value = {p_value: .4e}")

print("2.5 Getting training and testing sets")
df_x_pre: pd.DataFrame = df_prostate_encoded.drop(columns='lpsa')
df_y_pre: pd.DataFrame = df_prostate_encoded[['lpsa', 'train_T']]

x_train_pre: pd.DataFrame = df_x_pre[df_x_pre["train_T"]].drop(
    columns=["train_T"])
x_test_pre: pd.DataFrame = df_x_pre[~df_x_pre["train_T"]].drop(
    columns=["train_T"])
y_train_pre: pd.DataFrame = df_y_pre[df_y_pre["train_T"]]['lpsa'].drop(
    columns=["train_T"])
y_test_pre: pd.DataFrame = df_y_pre[~df_y_pre["train_T"]]['lpsa'].drop(
    columns=["train_T"])

print("2.6 Getting only numeric features of train set to "
      "apply Standard Scaler later")
x_train_pre_numeric_columns: pd.Index = \
    x_train_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_train_pre_numeric_columns)
print("2.7 Getting non-numeric features, in this case all boolean features")
x_train_pre_non_numeric_cols: pd.Index = x_train_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_train_pre_non_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_non_numeric_cols]
df_x_train_pre_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_numeric_columns]
print("")

print("2.8 Applying standard scaler to numeric features in train set")
std_sclr: StandardScaler = StandardScaler()
std_sclr.fit(df_x_train_pre_numeric)
x_train_numeric_scaled: np.ndarray = std_sclr.transform(df_x_train_pre_numeric)

print("2.9 Binding non numeric features with scaled numeric features again")
df_x_train_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_train_numeric_scaled,
    columns=x_train_pre_numeric_columns,
    index=x_train_pre[x_train_pre_numeric_columns].index
)

df_x_train_scaled: pd.DataFrame = pd.concat([df_x_train_numeric_scaled,
                                             df_x_train_pre_non_numeric],
                                            axis=1)
print("2.10 Getting only numeric features of test set to "
      "apply Standard Scaler later")
x_test_pre_numeric_columns: pd.Index = \
    x_test_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_test_pre_numeric_columns)
print("2.11 Getting non-numeric features, in this case all boolean features")
x_test_pre_non_numeric_cols: pd.Index = x_test_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_test_pre_non_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_non_numeric_cols]
df_x_test_pre_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_numeric_columns]

print("2.12 Applying Standard scale already trained for train set")
x_test_numeric_scaled: np.ndarray = std_sclr.transform(df_x_test_pre_numeric)

print("2.13 Binding non numeric features with scaled numeric features again")
df_x_test_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_test_numeric_scaled,
    columns=x_test_pre_numeric_columns,
    index=x_test_pre[x_test_pre_numeric_columns].index
)

df_x_test_scaled: pd.DataFrame = pd.concat([df_x_test_numeric_scaled,
                                            df_x_test_pre_non_numeric],
                                           axis=1)

x_train: pd.DataFrame = df_x_train_scaled
x_test: pd.DataFrame = df_x_test_scaled
y_train: pd.DataFrame = y_train_pre
y_test: pd.DataFrame = y_test_pre

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
y_test_plot: pd.Series = y_test.reset_index(drop=True)
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

print("5. Post-processing")
print("5.1 Getting model's coefficients")
linear_coefficients: np.ndarray = lm.coef_
ridge_coefficients: np.ndarray = lm_rid.coef_
lasso_coefficients: np.ndarray = lm_lasso.coef_

print("5.2 Comparing Lasso and Ridge coefficients to Linear")
models_coefficients_: dict = {"Linear": linear_coefficients,
                              "Ridge": ridge_coefficients,
                              "Lasso": lasso_coefficients}
colours_: [str] = ["red", "blue", "green"]

create_plot_compare_coefficients(models_coefficients_, colours_)

print("5.3 Applying Lasso reduction, removing every feature that is 0 "
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
col_selected_features = x_train.columns[
    df_lasso_coef["coefficients"].to_numpy() != 0]
df_filtered_data = df_prostate_encoded[col_selected_features]

print("6. Post processing data again")
print("6.1 Getting training and testing sets")
df_x_pre: pd.DataFrame = pd.concat([df_filtered_data,
                                    df_prostate_encoded[["train_T"]]], axis=1)
df_y_pre: pd.DataFrame = df_prostate_encoded[['lpsa', 'train_T']]
x_train_pre: pd.DataFrame = df_x_pre[df_x_pre["train_T"]].drop(
    columns=["train_T"])
x_test_pre: pd.DataFrame = df_x_pre[~df_x_pre["train_T"]].drop(
    columns=["train_T"])
y_train_pre: pd.DataFrame = df_y_pre[df_y_pre["train_T"]]['lpsa'].drop(
    columns=["train_T"])
y_test_pre: pd.DataFrame = df_y_pre[~df_y_pre["train_T"]]['lpsa'].drop(
    columns=["train_T"])

print("6.2 Getting only numeric features of train set to "
      "apply Standard Scaler later")
x_train_pre_numeric_columns: pd.Index = \
    x_train_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_train_pre_numeric_columns)
print("6.3 Getting non-numeric features, in this case all boolean features")
x_train_pre_non_numeric_cols: pd.Index = x_train_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_train_pre_non_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_non_numeric_cols]
df_x_train_pre_numeric: pd.DataFrame = x_train_pre[
    x_train_pre_numeric_columns]
print("")

print("6.4 Applying standard scaler to numeric features in train set")
std_sclr: StandardScaler = StandardScaler()
std_sclr.fit(df_x_train_pre_numeric)
x_train_numeric_scaled: np.ndarray = std_sclr.transform(df_x_train_pre_numeric)

print("6.5 Binding non numeric features with scaled numeric features again")
df_x_train_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_train_numeric_scaled,
    columns=x_train_pre_numeric_columns,
    index=x_train_pre[x_train_pre_numeric_columns].index
)

df_x_train_scaled: pd.DataFrame = pd.concat([df_x_train_numeric_scaled,
                                             df_x_train_pre_non_numeric],
                                            axis=1)
print("6.6 Getting only numeric features of test set to "
      "apply Standard Scaler later")
x_test_pre_numeric_columns: pd.Index = \
    x_test_pre.select_dtypes(include=['int64', 'float64']).columns
print(x_test_pre_numeric_columns)
print("6.7 Getting non-numeric features, in this case all boolean features")
x_test_pre_non_numeric_cols: pd.Index = x_test_pre. \
    select_dtypes(exclude=["int64", "float64"]).columns
df_x_test_pre_non_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_non_numeric_cols]
df_x_test_pre_numeric: pd.DataFrame = x_test_pre[
    x_test_pre_numeric_columns]

print("6.8 Applying Standard scale already trained for train set")
x_test_numeric_scaled: np.ndarray = std_sclr.transform(df_x_test_pre_numeric)

print("6.9 Binding non numeric features with scaled numeric features again")
df_x_test_numeric_scaled: pd.DataFrame = pd.DataFrame(
    x_test_numeric_scaled,
    columns=x_test_pre_numeric_columns,
    index=x_test_pre[x_test_pre_numeric_columns].index
)

df_x_test_scaled: pd.DataFrame = pd.concat([df_x_test_numeric_scaled,
                                            df_x_test_pre_non_numeric],
                                           axis=1)

x_train_sel: pd.DataFrame = df_x_train_scaled
x_test_sel: pd.DataFrame = df_x_test_scaled
y_train_sel: pd.DataFrame = y_train_pre
y_test_sel: pd.DataFrame = y_test_pre

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
y_pred_linear_sel_series = pd.Series(y_pred_linear_sel)
y_pred_ridge_sel_series = pd.Series(y_pred_ridge_sel)
y_pred_lasso_sel_series = pd.Series(y_pred_lasso_sel)
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
