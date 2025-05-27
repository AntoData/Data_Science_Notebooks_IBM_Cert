import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, \
    cross_val_predict
from sklearn.datasets import load_wine

"""
CORRELATION: 
Measures how dependent a variable is on a variable or group of variables

Pearson correlation:
- Correlation coefficient: Measures how dependent a variable is on 
    another variable: 
    -   1=Large positive correlation
    -   0=No correlation
    -   -1=Large negative correlation
- p-value: How accurate correlation coefficient is:
    -   p < 0.001 = Strong certainty
    -   p < 0.005 = Moderate certainty
    -   p < 0.1 = Weak certainty
    -   p > 0.1 = No certainty  
"""

# First we load a example dataset provided by sklearn
wine_data = load_wine(as_frame=True)
df_wine: pd.DataFrame = wine_data.frame

print("Columns: {0}".format(df_wine.columns))
print("Column types: {0}".format(df_wine.dtypes))

# We can get correlation coefficients between columns: df.corr()
i_best: int = -1
j_best: int = -1
max_corr_coef: float = 0
p_value_corr: float = 10
print("Analysing correlation")
df_correlation_matrix_wine: pd.DataFrame = df_wine.corr()
for i in range(0, df_correlation_matrix_wine.shape[0]):
    for j in range(0, df_correlation_matrix_wine.shape[1]):
        if i != j:
            print("{0}, {1} = {2}".format(
                df_correlation_matrix_wine.columns[i],
                df_correlation_matrix_wine.columns[j],
                df_correlation_matrix_wine.iloc[i, j]))
            if abs(df_correlation_matrix_wine.iloc[i, j]) > abs(max_corr_coef):
                # Now we need to get how accurate the correlation
                # In order to do so we use stats.pearsonr
                coef_corr, p_value = stats.pearsonr(
                    df_wine[df_correlation_matrix_wine.columns[i]],
                    df_wine[df_correlation_matrix_wine.columns[j]])
                print("Corr coef: {0}".format(coef_corr))
                print("p-value: {0}".format(p_value))
                # If the p_value shows strong certainty
                if 0.001 >= p_value:
                    p_value_corr = p_value
                    # coefficient is using: scipy.stats.pearsonr
                    print("New best correlation")
                    max_corr_coef = df_correlation_matrix_wine.iloc[i, j]
                    i_best = i
                    j_best = j
# Best correlation is:
print("")
print("Best correlation")
print("{0}, {1} = {2}".format(df_wine.columns[i_best], df_wine.columns[j_best],
                              max_corr_coef))
x_column_name: str = df_wine.columns[i_best]
y_column_name: str = df_wine.columns[j_best]
print("p-value={0}".format(p_value_corr))

"""
MODELS: Mathematical equations used to predict a value given one or 
more values -> Shape in which the correlation takes place in
"""
# We will start with SIMPLE LINEAR REGRESSION
# 1. Train model
lr = LinearRegression()
lr.fit(df_wine[[x_column_name]], df_wine[y_column_name])

# 2. Predict
y_pred = lr.predict(df_wine[[x_column_name]])
plt.plot(df_wine[x_column_name], y_pred)
plt.scatter(df_wine[x_column_name], df_wine[y_column_name])
plt.title("{0} vs {1}".format(x_column_name, y_column_name))
plt.xlabel(x_column_name)
plt.ylabel(y_column_name)
plt.show()

"""
y=b0 + b1*x we can get b0 and b1
"""
lr_intercerpt: float = lr.intercept_
lr_coef: float = lr.coef_
print("y={0} + {1}*x".format(lr_intercerpt, lr_coef))
print("")

"""
PROBLEM: Correlation coefficient tells us how dependent a variable is 
on another group of variables but not how well the model fits the data
used to train the model

SOLUTION:
    - Mean Squared Error: The lower the better
    - R^2: 1 is good, 0 is bad
"""
# We can get R^2 from the model using LinearRegression.score(x,y)
lr_r2: float = lr.score(df_wine[[x_column_name]], df_wine[y_column_name])
print("R^2={0}".format(lr_r2))
if lr_r2 > 0.90:
    print("Good R^2")
elif lr_r2 > 0.75:
    print("Acceptable R^2")
else:
    print("Not very good R^2")
print("")
# We can get mse using mean_squared_error from sklearn.metrics
lr_mse: float = mean_squared_error(df_wine[y_column_name], y_pred)
lr_r2_: float = r2_score(df_wine[y_column_name], y_pred)
print("mse={0}, r^2={1}".format(lr_mse, lr_r2_))

# We will continue with MULTIPLE LINEAR REGRESSION
# The other variable with a high correlation with y is target
# So independent variables are total_phenols, target (x1, x2) and
# dependent variable is flavanoids (y)
print("MULTIPLE LINEAR REGRESSION")
print("flavanoids = b0 + b1*total_phenols + b2*target")
# 1. Train model
mlr = LinearRegression()
mlr.fit(df_wine[["total_phenols", "target"]], df_wine["flavanoids"])
mlr_b0 = mlr.intercept_
mlr_b1 = mlr.coef_
print("flavanoids = {0} + {1}*total_phenols + {2}*target".format(mlr_b0,
                                                                 mlr_b1[0],
                                                                 mlr_b1[1]))
# 2. Predict
mlr_y_pred = mlr.predict(df_wine[["total_phenols", "target"]])

# mse, r^2
mlr_r2: float = mlr.score(df_wine[["total_phenols", "target"]],
                          df_wine["flavanoids"])
print("R^2={0}".format(mlr_r2))
if mlr_r2 > 0.90:
    print("Good R^2")
elif mlr_r2 > 0.75:
    print("Acceptable R^2")
else:
    print("Not very good R^2")
print("This model adjusts better than Simple Linear Regression")
mlr_mse: float = mean_squared_error(df_wine["flavanoids"], mlr_y_pred)
mlr_r2: float = r2_score(df_wine["flavanoids"], mlr_y_pred)
print("mse={0}, r^2={1}".format(mlr_mse, mlr_r2))

# We will continue with POLYNOMIAL LINEAR REGRESSION
# We will pick degree 3
print("POLYNOMIAL LINEAR REGRESSION")
print("flavanoids = b0 + b1*total_phenols + b2*total_phenols^2 + "
      "b3*total_phenols^3")
# 1. Transform x to polynomial features
poly = PolynomialFeatures(degree=3)
pr_3_x = poly.fit_transform(df_wine[[x_column_name]])

# 2. Train the linear model
plm = LinearRegression()
plm.fit(pr_3_x, df_wine[y_column_name])

plm_b0 = plm.intercept_
plm_bx = plm.coef_
print(plm_bx)
print("flavanoids = {0} + {1}*total_phenols + {2}*total_phenols^2 + "
      "{3}*total_phenols^3".format(plm_b0, plm_bx[1], plm_bx[2], plm_bx[3]))

# 3. Predict
plm_y_pred = plm.predict(pr_3_x)

# Graphic visualization
plt.scatter(df_wine[x_column_name], plm_y_pred)
plt.scatter(df_wine[x_column_name], df_wine[y_column_name])
plt.title("{0} vs {1}".format(x_column_name, y_column_name))
plt.xlabel(x_column_name)
plt.ylabel(y_column_name)
plt.show()

# 4. Metrics
plr_3_r2: float = plm.score(pr_3_x, df_wine["flavanoids"])
print("R^2={0}".format(plr_3_r2))
if plr_3_r2 > 0.90:
    print("Good R^2")
elif plr_3_r2 > 0.75:
    print("Acceptable R^2")
else:
    print("Not very good R^2")
print("R^2 is worse using a 3rd degree polynomial regression")
plr_mse: float = mean_squared_error(df_wine["flavanoids"], plm_y_pred)
plr_r2: float = r2_score(df_wine["flavanoids"], plm_y_pred)
print("mse={0}, r^2={1}".format(plr_mse, plr_r2))

# With numpy
# 1. Train model
f_pr = np.polyfit(df_wine[x_column_name], df_wine[y_column_name], 3)
exp_pr = np.poly1d(f_pr)
print(exp_pr)

# 2. Predict
f_pr_y_pred = np.polyval(f_pr, df_wine[x_column_name])
plt.scatter(df_wine[x_column_name], f_pr_y_pred)
plt.scatter(df_wine[x_column_name], df_wine[y_column_name])
plt.title("{0} vs {1}".format(x_column_name, y_column_name))
plt.xlabel(x_column_name)
plt.ylabel(y_column_name)
plt.show()

# We will continue with POLYNOMIAL MULTILINEAR REGRESSION
# We will pick degree 2

# 1. We transform x to polynomial features
mpoly = PolynomialFeatures(degree=2)
x_mpr = mpoly.fit_transform(df_wine[["total_phenols", "target"]])

# 2. We train the model
pmlr = LinearRegression()
pmlr.fit(x_mpr, df_wine["flavanoids"])

# 3. Predictions
y_pred_mpr = pmlr.predict(x_mpr)

# 4. Metrics
pmlr_2_r2 = pmlr.score(x_mpr, df_wine["flavanoids"])
print("R^2={0}".format(plr_3_r2))
if pmlr_2_r2 > 0.90:
    print("Good R^2")
elif pmlr_2_r2 > 0.75:
    print("Acceptable R^2")
else:
    print("Not very good R^2")
mplr_mse: float = mean_squared_error(df_wine["flavanoids"], plm_y_pred)
mplr_r2: float = r2_score(df_wine["flavanoids"], plm_y_pred)
print("mse={0}, r^2={1}".format(mplr_mse, mplr_r2))

"""
PROBLEM: MSE and R^2 determine how well our data fits the model, but 
won't tell us how well it predicts data

SOLUTION:
    - We split the data in two sets:
        - One for training
        - One for testing
"""
# We will repeat MULTIPLE LINEAR REGRESSION and POLYNOMIAL MULTILINEAR
# REGRESSION using two different sets of variables for training and
# testing

# MULTIPLE LINEAR REGRESSION
# So independent variables are total_phenols, target (x1, x2) and
# dependent variable is flavanoids (y)

# 1. First, we create the training and testing data sets using
# train_test_split, 60% will be train datasets

x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(
    df_wine[["total_phenols", "target"]], df_wine["flavanoids"],
    train_size=0.6)

# 2. Train the model using x_train_lr and y_train_lr
split_lr = LinearRegression()
split_lr.fit(x_train_lr, y_train_lr)
print("Intercept={0}".format(split_lr.intercept_))
print("Coeficient={0}".format(split_lr.coef_))
print("y={0} + {1}*x1 + {2}*x2".format(split_lr.intercept_,
                                       split_lr.coef_[0],
                                       split_lr.coef_[1]))
# 3. We use x_test_lr to predict values
y_test_pred_lr = split_lr.predict(x_test_lr)

# 4. We use now y_test_pred_lr and y_test_lr to work out mse and r^2
mse_split_lr = mean_squared_error(y_test_lr, y_test_pred_lr)
r2_split_lr = r2_score(y_test_lr, y_test_pred_lr)
print("mse={0}, r^2={1}".format(mse_split_lr, r2_split_lr))
print("not split mse={0}, split mse={1}".format(mlr_mse, mse_split_lr))
print("not split r^2={0}, split r^2={1}".format(mlr_r2, r2_split_lr))

# Let's repeat with different samples

# 1. First, we create the training and testing data sets using
# train_test_split, 60% will be train datasets

x_train_lr2, x_test_lr2, y_train_lr2, y_test_lr2 = train_test_split(
    df_wine[["total_phenols", "target"]], df_wine["flavanoids"],
    train_size=0.6)

# 2. Train the model using x_train_lr and y_train_lr
split_lr2 = LinearRegression()
split_lr2.fit(x_train_lr2, y_train_lr2)
print("Intercept={0}".format(split_lr2.intercept_))
print("Coeficient={0}".format(split_lr2.coef_))
print("y={0} + {1}*x1 + {2}*x2".format(split_lr2.intercept_,
                                       split_lr2.coef_[0],
                                       split_lr2.coef_[1]))
# 3. We use x_test_lr to predict values
y_test_pred_lr2 = split_lr2.predict(x_test_lr2)

# 4. We use now y_test_pred_lr and y_test_lr to work out mse and r^2
mse_split_lr2 = mean_squared_error(y_test_lr2, y_test_pred_lr2)
r2_split_lr2 = r2_score(y_test_lr2, y_test_pred_lr2)
print("mse={0}, r^2={1}".format(mse_split_lr2, r2_split_lr2))
print("First mse={0}, second mse={1}".format(mse_split_lr, mse_split_lr2))
print("First r^2={0}, second r^2={1}".format(r2_split_lr, r2_split_lr2))
# train_test_split introduces a random aspect to the model as different
# samples produce different results

# We will continue with MULTIPLE LINEAR REGRESSION
# The other variable with a high correlation with y is target
# So independent variables are total_phenols, target (x1, x2) and
# dependent variable is flavanoids (y)

# 1. Split the dataset in a training sample and testing sample

x_train, x_test, y_train, y_test = \
    train_test_split(
        df_wine[["total_phenols", "target"]], df_wine["flavanoids"],
        train_size=0.6)

# 2. We transform X to polynomial features
pr_d2 = PolynomialFeatures(degree=2)
x_train_mpr = pr_d2.fit_transform(x_train)
x_test_mpr = pr_d2.fit_transform(x_test)

# 3. We train the linear model using x_train_mpr
pr_lr = LinearRegression()
pr_lr.fit(x_train_mpr, y_train)

# 4. We predict the values in x_test_mpr and we will use them
# to get mse and r^2
y_test_mpr_pred = pr_lr.predict(x_test_mpr)

# 5. Getting mse, r^2
r2_mpr = pr_lr.score(x_test_mpr, y_test)
print("r^2={0}".format(r2_mpr))
r2_mpr_ = r2_score(y_test, y_test_mpr_pred)
mse_mpr = mean_squared_error(y_test, y_test_mpr_pred)
print("mse={0}, r^2={1}".format(mse_mpr, r2_mpr_))

# If we repeat the same process
# 1. Split the dataset in a training sample and testing sample

x_train2, x_test2, y_train2, y_test2 = \
    train_test_split(
        df_wine[["total_phenols", "target"]], df_wine["flavanoids"],
        train_size=0.6)

# 2. We transform X to polynomial features
pr_d2_ = PolynomialFeatures(degree=2)
x_train_mpr2 = pr_d2_.fit_transform(x_train2)
x_test_mpr2 = pr_d2_.fit_transform(x_test2)

# 3. We train the linear model using x_train_mpr
pr_lr2_ = LinearRegression()
pr_lr2_.fit(x_train_mpr2, y_train2)

# 4. We predict the values in x_test_mpr and we will use them
# to get mse and r^2
y_test_mpr_pred2 = pr_lr2_.predict(x_test_mpr2)

# 5. Getting mse, r^2
r2_mpr2 = pr_lr2_.score(x_test_mpr2, y_test2)
print("r^2={0}".format(r2_mpr2))
r2_mpr2_ = r2_score(y_test2, y_test_mpr_pred2)
mse_mpr2 = mean_squared_error(y_test2, y_test_mpr_pred2)
print("mse={0}, r^2={1}".format(mse_mpr2, r2_mpr2_))
print("First mse={0}, second mse={1}".format(mse_mpr, mse_mpr2))
print("First r^2={0}, second r^2={1}".format(r2_mpr, r2_mpr2))

# As we can see the results depend on the samples we get

"""
PROBLEM: Sometimes the difference in scales between variables might make
these differences even greater. One of the independent variables because
of difference in scale might make the difference between training and 
testing samples even greater
"""

"""
SOLUTION: Standarize and scale the variables
In order to do so we use StandardScaler
"""
# MULTIPLE LINEAR REGRESSION EXAMPLE
# 1. We divide the dataset between training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
    df_wine[["total_phenols", "target"]], df_wine["flavanoids"],
    train_size=0.6)

# 2. We use StandardScaler to scale the x datasets
scaler = StandardScaler()
scaler.fit(x_train)  # IMPORTANT: We use x_train to train the scaler
# Now we scale the datasets for independent variables (X)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 3. Now we train the linear model with these scaled datasets for x
l_model = LinearRegression()
l_model.fit(x_train_scaled, y_train)

# 4. We predict values using our scaled test dataset
y_pred_test = l_model.predict(x_test_scaled)

# 5. We get MSE and R^2
mse_scaled = mean_squared_error(y_test, y_pred_test)
r2_scaled = r2_score(y_test, y_pred_test)
print("mse={0}, r^2={1}".format(mse_scaled, r2_scaled))
# Again different MSE and R^2

# MULTIPLE POLYNOMIAL EXAMPLE
# 1. We divide our dataset in training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df_wine[["total_phenols", "target"]], df_wine["flavanoids"],
    train_size=0.6)

# 2. We transform our X datasets to polynomial feature form
poly_2_model = PolynomialFeatures(degree=2)
x_train_poly_2 = poly_2_model.fit_transform(x_train)
x_test_poly_2 = poly_2_model.fit_transform(x_test)

# 3. We scale them using StandardScaler
scaler_p = StandardScaler()
scaler_p.fit(x_train_poly_2, y_train)
x_train_scaled_poly_2 = scaler_p.transform(x_train_poly_2)
x_test_scaled_poly_2 = scaler_p.transform(x_test_poly_2)

# 4. We train our linear model
l_model_sc = LinearRegression()
l_model_sc.fit(x_train_scaled_poly_2, y_train)

# 5. We use the model to predict the values in test dataset for x
y_test_pred_p2 = l_model_sc.predict(x_test_scaled_poly_2)

# 6. We use the prediction to work out MSE and R^2
mse_sc_p2 = mean_squared_error(y_test, y_test_pred_p2)
r2_sc_p2 = r2_score(y_test, y_test_pred_p2)
print("mse={0}, r^2={1}".format(mse_sc_p2, r2_sc_p2))
# Again different MSE and R^2

"""
PROBLEM: Difference in the datasets keep making our models different so
there is a random factor we cannot control that might change our model
and our metrics to see if the model is good at predicting values or not
"""

"""
SOLUTION: CROSS VALIDATION
We split the data in "K" folds (groups)
-   Some will be used for training the model
-   Some will be used for testing our model
But now we will repeat the process until each sample has been used for
training and testing
Avg is the result of out-of-sample error
We use cross_val_score to get "k" r^2 values (?)
We use cross_val_predict to get "k" groups of predictions using the same
model with different training and testing datasets (?)
"""
# MULTIPLE LINEAR REGRESSION EXAMPLE
# 1. We create and train the model
lin_model = LinearRegression()

# 2. We use cross_val_score to validate how well the model works
df_wine_shuffled = df_wine.sample(frac=1)  # We should shuffle data bef
r2_scores = cross_val_score(lin_model,
                            df_wine_shuffled[["total_phenols", "target"]],
                            df_wine_shuffled["flavanoids"], cv=4)
print("R^2s = {0}".format(r2_scores))
print("Mean R^2 = {0}".format(r2_scores.mean()))

# 3. We can use cross_val_predict to predict values
y_pred_cv = cross_val_predict(
    lin_model, df_wine_shuffled[["total_phenols", "target"]],
    df_wine_shuffled["flavanoids"], cv=4)

print("Predictions")
print(y_pred_cv)

# 4. We get the accuracy (R^2)
r2_cv = r2_score(df_wine_shuffled["flavanoids"], y_pred_cv)
print("R^2 = {0}".format(r2_cv))

# MULTIPLE POLYNOMIAL REGRESSION EXAMPLE
# 1. We turn the X dataset into polynomial features
poly_3d = PolynomialFeatures(degree=3)
x_poly_3 = poly_3d.fit_transform(df_wine_shuffled[["total_phenols", "target"]])

# 2. We create the Linear model
lp3_model = LinearRegression()

# 3. We get cross validation score
scores_cv_3p = cross_val_score(lp3_model, x_poly_3,
                               df_wine_shuffled["flavanoids"])
print("R^2s = {0}".format(scores_cv_3p))
print("Mean R^2 = {0}".format(scores_cv_3p.mean()))

# 4. We use cross validation to predict values
y_pred_cv_3p = cross_val_predict(lp3_model, x_poly_3,
                                 df_wine_shuffled["flavanoids"])
print("Predictions = {0}".format(y_pred_cv_3p))

# 5. We get accuracy
accuracy = r2_score(df_wine_shuffled["flavanoids"], y_pred_cv_3p)
print("Accuracy = {0}".format(accuracy))

"""
PROBLEM: How do we pick a model? Models can be:
    - Underfitting: The curve is too simple to fit the data
    - Overfitting: Too flexible, it fits noise rather than data
"""

"""
SOLUTION: Iteration over degree complexity
"""
# MULTIPLE POLYNOMIAL REGRESSION EXAMPLE using train sets and test sets
# 1. We divide the dataset in training and testing groups

df_wine_shuffled = df_wine.sample(frac=1)  # Always shuffle sample before
x_train, x_test, y_train, y_test = \
    train_test_split(df_wine_shuffled[["total_phenols", "target"]],
                     df_wine_shuffled["flavanoids"], train_size=0.6)

results_order_scores: dict = {}
results_order_predictions: dict = {}
# 2. We iterate through different polynomial orders
degrees = [i for i in range(1, 7)]

for degree_ in degrees:
    print("Trying with degree = {0}".format(degree_))

    # 3. We transform our X training and testing sets to polynomial form
    poly_it = PolynomialFeatures(degree=degree_)
    x_train_poly_it = poly_it.fit_transform(x_train)
    x_test_poly_it = poly_it.fit_transform(x_test)

    # 4. We train now our linear model
    linear_m_it = LinearRegression()
    linear_m_it.fit(x_train_poly_it, y_train)

    # 5. We get the score for this order and save it
    score_ = linear_m_it.score(x_test_poly_it, y_test)
    results_order_scores[degree_] = score_

    # 6. We use the model to predict values and save them
    y_test_pred_it = linear_m_it.predict(x_test_poly_it)
    results_order_predictions[degree_] = y_test_pred_it

print("Let's compare scores")
for degree_ in results_order_scores.keys():
    print("Order = {0} - R^2 = {1}".format(degree_,
                                           results_order_scores[degree_]))

# If we execute this loop several times we will see the higher the
# degree the more volatile R^2 becomes. A change in samples makes R^2
# much higher or lower

# MULTIPLE POLYNOMIAL REGRESSION EXAMPLE using cross validation
# Against degrees from 1 to 6 and using k=4 or k = 5
results_cross_score: dict = {}
results_cross_predictions: dict = {}
degrees_cv = [i for i in range(1, 7)]
k_folds = [4, 5]

df_wine_shuffled = df_wine.sample(frac=1)  # Always shuffle sample before
y = df_wine_shuffled["flavanoids"]

# 1. We iterate through the different degrees
for degree_ in degrees_cv:
    print("Trying with degree = {0}".format(degree_))
    # 2. We transform x to Polynomial feature form
    poly_it_cv = PolynomialFeatures(degree=degree_)
    x_poly_it_cv = \
        poly_it_cv.fit_transform(df_wine_shuffled[["total_phenols", "target"]])

    # 3. We get the linear model
    lin_model_cv = LinearRegression()

    # 4. We iterate through the ks
    for k_fold_ in k_folds:
        # 4.1 - We use cross_val_score to get the scores
        scores_ = cross_val_score(lin_model_cv, x_poly_it_cv, y, cv=k_fold_)
        results_cross_score["{0}-{1}".format(degree_, k_fold_)] = scores_

        # 4.2 - We use cross val_predict to cross validate predictions
        preds_ = cross_val_predict(lin_model_cv, x_poly_it_cv, y, cv=k_fold_)
        results_cross_predictions[degree_] = preds_

for degree_fold in results_cross_score.keys():
    print("degree, k-folds = {0}, R^2 scores = {1}".format(
        degree_fold, results_cross_score[degree_fold]))

# PROBLEM: Models with multiple linear variables and ones with
# polynomial features have colinear combinations of features which
# cause OVERFITTING

# SOLUTION: Regularize feature sets using hyperparameters, in this
# case alpha

"""
RIDGE REGRESSION: Controls the order of polynomial coefficients
Ridge object replaces object LinearRegression
"""
# Lineal model
# 1. We create a Ridge Regression model object and set alpha
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1)

# 2. We train the model
ridge_model.fit(df_wine_shuffled[["total_phenols", "target"]],
                df_wine_shuffled["flavanoids"])

# 3. We predict
y_rm_pred = ridge_model.predict(df_wine_shuffled[["total_phenols", "target"]])

# 4. We use the prediction to get the MSE and R^2 scores
r2_score_rm = r2_score(df_wine_shuffled["flavanoids"], y_rm_pred)
mse_rm = mean_squared_error(df_wine_shuffled["flavanoids"], y_rm_pred)

print("r2 = {0}".format(r2_score_rm))
print("mse = {0}".format(mse_rm))

# MULTIPLE POLYNOMIAL RIDGE REGRESSION
# 1. We transform the dataset to Polynomial features
poly_rm_3 = PolynomialFeatures(degree=3)
x_poly_rm_3 = poly_rm_3.fit_transform(
    df_wine_shuffled[["total_phenols", "target"]])

# 2. We create the Ridge regression model and set alpha
poly_ridge_3 = Ridge(alpha=0.1)

# 3. We train the model
poly_ridge_3.fit(x_poly_rm_3, df_wine_shuffled["flavanoids"])

# 4. We use the model to predict values
y_poly_ridge_3_pred = poly_ridge_3.predict(x_poly_rm_3)

# 5. We use the prediction to get MSE and R^2
mse_poly_3_rm = mean_squared_error(df_wine_shuffled["flavanoids"],
                                   y_poly_ridge_3_pred)
r2_poly_3_rm = r2_score(df_wine_shuffled["flavanoids"],
                        y_poly_ridge_3_pred)
print("r2 = {0}".format(r2_poly_3_rm))
print("mse = {0}".format(mse_poly_3_rm))

# MULTIPLE POLYNOMIAL RIDGE REGRESSION PLUS CROSS VALIDATION
# 1. We transform X to polynomial features using a degree of 3
poly_3_cv_rm = PolynomialFeatures(degree=3)
x_poly3_cv_rm = poly_3_cv_rm.fit_transform(
    df_wine_shuffled[["total_phenols", "target"]])

# 2. We create the ridge model and set alpha
ridge_poly_3_cv_rm = Ridge(alpha=0.1)
ridge_poly_3_cv_rm.fit(x_poly3_cv_rm, df_wine_shuffled["flavanoids"])

# 3. We get cross validated scores
scores_poly3_cv_rm = cross_val_score(ridge_poly_3_cv_rm, x_poly3_cv_rm,
                                     df_wine_shuffled["flavanoids"], cv=4)
print("R^2 scores = {0}".format(scores_poly3_cv_rm))

# 4. We cross predict
y_poly3_cv_rm_pred = cross_val_predict(ridge_poly_3_cv_rm, x_poly3_cv_rm,
                                       df_wine_shuffled["flavanoids"], cv=4)

# 5. We use the predictions to get mean squared error
mse_poly3_cv_rm = mean_squared_error(df_wine_shuffled["flavanoids"],
                                     y_poly3_cv_rm_pred)
print("MSE = {0}".format(mse_poly3_cv_rm))

# MULTIPLE POLYNOMIAL RIDGE REGRESSION ITERATING THROUGH DEGREES
degrees_cv_rm = [x for x in range(1, 7)]
for degree_cv_rm in degrees_cv_rm:
    # 1. We transform X to polynomial feature
    poly_it_cv_rm = PolynomialFeatures(degree=degree_cv_rm)
    x_poly_it_cv_rm = poly_it_cv_rm.fit_transform(
        df_wine_shuffled[["total_phenols", "target"]])

    # 2. We create ridge regression model object and set alpha
    ridge_poly_it_cv_rm = Ridge(alpha=0.1)

    # 3. We get cross validated score
    r2_poly_it_cv_rm = cross_val_score(ridge_poly_it_cv_rm, x_poly_it_cv_rm,
                                       df_wine_shuffled["flavanoids"], cv=4)
    print("Degree = {0} - R^2 scores = {1}".format(degree_cv_rm,
                                                   r2_poly_it_cv_rm))

    # 4. We get cross validated predictions
    y_poly_it_cv_rm = cross_val_predict(ridge_poly_3_cv_rm, x_poly_it_cv_rm,
                                        df_wine_shuffled["flavanoids"], cv=4)

    # 5. We use the predictions to get MSE
    mse_poly_it_cv_rm = mean_squared_error(df_wine_shuffled["flavanoids"],
                                           y_poly_it_cv_rm)
    print("Degree = {0} - MSE = {1}".format(degree_cv_rm, mse_poly_it_cv_rm))

# MULTIPLE POLYNOMIAL RIDGE REGRESSION ITERATING THROUGH DEGREES AND
# DIFFERENT VALUES FOR ALPHA
degrees_cv_rm = [x for x in range(1, 7)]
alphas_cv_rm = [0.0001, 0.001, 0.01, 0.1, 1, 10]

for degree_cv_rm in degrees_cv_rm:
    # 1. We set degree and transform X to polynomial features
    poly_it_cv_rm_a = PolynomialFeatures(degree=degree_cv_rm)
    x_poly_it_cv_rm_a = poly_it_cv_rm_a.fit_transform(
        df_wine_shuffled[["total_phenols", "target"]])

    # 2. We iterate through alphas
    for alpha_cv_rm in alphas_cv_rm:
        # 3. We create our Ridge regression model object and set alpha
        ridge_poly_it_cv_rm_a = Ridge(alpha=alpha_cv_rm)

        # 4. We train the model
        ridge_poly_it_cv_rm_a.fit(x_poly_it_cv_rm_a,
                                  df_wine_shuffled["flavanoids"])

        # 5. We use the model to get a cross validated r^2 score
        scores_poly_it_cv_rm_a = cross_val_score(
            ridge_poly_it_cv_rm_a, x_poly_it_cv_rm_a,
            df_wine_shuffled["flavanoids"], cv=4)
        print("Degree = {0} - alpha = {1} - R^2 = {2}".format(
            degree_cv_rm, alpha_cv_rm, scores_poly_it_cv_rm_a))

        # 6. We use the model to get cross validated predictions
        y_poly_it_cv_rm_a_pred = cross_val_predict(
            ridge_poly_it_cv_rm_a, x_poly_it_cv_rm_a,
            df_wine_shuffled["flavanoids"], cv=4)

        # 7. We use prediction to get MSE
        mse_poly_it_cv_rm_a = mean_squared_error(
            df_wine_shuffled["flavanoids"], y_poly_it_cv_rm_a_pred)
        print("Degree = {0} - alpha = {1} - mse = {2}".format(
            degree_cv_rm, alpha_cv_rm, mse_poly_it_cv_rm_a))

# PROBLEM: What's the best value for alpha
# SOLUTION: Automatically iterate over hyperparameters using cross
# validation search

"""
GRID SEARCH: This automatically iterates over hyperparameters using 
cross validation search and can offer best models
Object GridSearch is the one that trains the model (uses .fit)
"""
from sklearn.model_selection import GridSearchCV

# 1. We create the dictionary with the alpha parameters we want to check
# parameters_gs = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 100],
#                  'normalize': [True, False]}
# The other parameter normalize using True and False will return for
# each value in alpha one result normalizing the data and another not
# normalizing the data but it is deprecated currently
parameters_gs = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 100]}

# 2. We need to create a Ridge (regression) object
ridge_gridsearch = Ridge()

# 3. We create a GridSearch object like this:
# This object replaces LinearRegression and Ridge Regression as the
# object to train and replaces train_test_split as it won't be necessary
grid_search = GridSearchCV(ridge_gridsearch, parameters_gs, cv=4)
# We are cross validating in 4-folds the different alpha parameters
# proposed for our ridge regression model

# 4. We train this Grid Search model
grid_search.fit(df_wine_shuffled[["total_phenols", "target"]],
                df_wine_shuffled["flavanoids"])

# 5. We can get the best estimator
best_estimator_gs = grid_search.best_estimator_

# 6. We can use this model to predict cross validated values
y_cv_grid_search_pred = grid_search.predict(
    df_wine_shuffled[["total_phenols", "target"]])
print("Predictions = {0}".format(y_cv_grid_search_pred))

# 7. We can get the cross validated scores
scores_grid_search = grid_search.cv_results_
print("scores = {0}".format(scores_grid_search))

# 8. We can use the best estimator to get R^2
score_grid_search_best = best_estimator_gs.score(
    df_wine_shuffled[["total_phenols", "target"]],
    df_wine_shuffled["flavanoids"])
print("Best R^2 = {0}".format(score_grid_search_best))

# 9. We can use the best estimator to predict our values
y_best_estimator_gs_pred = \
    best_estimator_gs.predict(df_wine_shuffled[["total_phenols", "target"]])
print("Best predictions: = {0}".format(y_best_estimator_gs_pred))

# Example iterating through degrees
# 1. We create a dictionary that will contain all our possible alphas
parameters_gs_di = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
degrees_gs_di = [x for x in range(1, 7)]

# 2. We iterate through degrees
for degree_gs_di in degrees_gs_di:
    print("Degree = {0}".format(degree_gs_di))
    # 2.1 We convert X to polynomial features using this degree
    poly_gs_di = PolynomialFeatures(degree=degree_gs_di)
    x_poly_gs_di = poly_gs_di.fit_transform(df_wine_shuffled[["total_phenols",
                                                              "target"]])

    # 3. We build the Ridge Regression object
    ridge_poly_ds_di = Ridge()

    # 4. We build our GridSearchCV object
    grid_search_gs_di = GridSearchCV(ridge_poly_ds_di, parameters_gs_di, cv=4)

    # 5. We train our model using X in polynomial form
    grid_search_gs_di.fit(x_poly_gs_di, df_wine_shuffled["flavanoids"])

    # 6. We can get cross validated R^2 scores
    scores_gs_di = grid_search_gs_di.cv_results_
    print("Degree = {0} - scores = {1}".format(degree_gs_di, scores_gs_di))

    # 7. We can cross predict values
    y_pred_gs_di = grid_search_gs_di.predict(x_poly_gs_di)
    print("Degree = {0} - General predictions = {1}".format(degree_gs_di,
                                                            y_pred_gs_di))

    # 8. We can use this to get MSE
    mse_gs_di = mean_squared_error(df_wine_shuffled["flavanoids"],
                                   y_pred_gs_di)
    print("Degree = {0} - General predictions = {1}".format(degree_gs_di,
                                                            mse_gs_di))

    # 9. We get the best estimator
    best_estimator_gs_di = grid_search_gs_di.best_estimator_

    # 10. We can use this best estimator to get its score
    scores_best_gs_di = best_estimator_gs_di.score(
        x_poly_gs_di, df_wine_shuffled["flavanoids"])
    print("Degree = {0} - Best score = {1}".format(
        degree_gs_di, scores_best_gs_di))

    # 11. We can use this best estimator to predict values
    y_best_estimator_gs_di_pred = best_estimator_gs_di.predict(x_poly_gs_di)
    print("Degree = {0} - Best estimator predictions = {0}".format(
        degree_gs_di, y_best_estimator_gs_di_pred))

    # 12. We can use this to get MSE
    mse_best_gs_di = mean_squared_error(df_wine_shuffled["flavanoids"],
                                        y_best_estimator_gs_di_pred)
    print("Degree = {0} - Best estimator MSE = {1}".format(degree_gs_di,
                                                           mse_best_gs_di))

count, bin_edges = np.histogram(df_wine_shuffled["flavanoids"], 10)
plt.hist(df_wine_shuffled["flavanoids"], bins=10)
plt.xticks(bin_edges)
plt.show()
