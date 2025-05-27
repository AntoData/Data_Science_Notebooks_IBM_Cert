import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

"""
SOURCE: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
"""
# 1. First we open the file and explore its data
degrees_scores: dict = {}
print("1. Opening the file")
df_life_expectancy: pd.DataFrame = pd.read_csv('Life Expectancy Data.csv')
print("1.1 Dropping NA rows")
df_life_expectancy.dropna(inplace=True)
print("2. Getting correlation matrix")
corr_matrix = df_life_expectancy.corr(numeric_only=True)
print(corr_matrix)
print("")
print("2.1 Getting only correlation to variable Life expectancy")
y_data_col: str = "Life expectancy "
corr_life_expectancy = corr_matrix[[y_data_col]]
print(corr_life_expectancy)
print("")
print("2.2 Getting all possible combination of two columns "
      "(except Life expentancy)")
df_life_expectancy_col: pd.DataFrame = corr_matrix.drop(
    columns=['Life expectancy ', 'Year'])
column_pairs = combinations(list(df_life_expectancy_col.columns), 2)
print("")
for column_pair in column_pairs:
    print("3. Applying standard scaler")
    col1 = column_pair[0]
    col2 = column_pair[1]
    std_scaler = StandardScaler()
    std_scaler.fit(df_life_expectancy[[col1, col2]],
                   df_life_expectancy[y_data_col])
    x_scaled = std_scaler.transform(df_life_expectancy[[col1, col2]])

    # 4. Applying Polynomial Transformation
    print("4. Applying the polynomial transformation")
    degrees = [degree for degree in range(2, 11)]
    print("Degrees = {0}".format(degrees))
    alpha_values: dict = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

    for degree in degrees:
        print(
            "4.1 Applying the polynomial transformation for degree = "
            "{0}".format(degree))
        poly_feat_dx = PolynomialFeatures(degree=degree)
        x_poly_scaled = poly_feat_dx.fit_transform(x_scaled)
        print("")

        print("4.2 Creating Ridge Model")
        ridge_model = Ridge()
        print("")

        print("4.3 Creating and training GridSearchCV")
        grid_search_cv = GridSearchCV(ridge_model, alpha_values, cv=5)
        print("")

        print("4.4 Training the model")
        grid_search_cv.fit(x_poly_scaled, df_life_expectancy[y_data_col])
        print("")

        print("4.5 Getting Grid Search CV scores")
        scores_gs = grid_search_cv.cv_results_
        print("scores = {0}".format(scores_gs))
        print("")

        print("4.6 Predicting y using x")
        y_pred = grid_search_cv.predict(x_poly_scaled)
        print(y_pred)
        print("")

        print("4.7 Getting best estimator")
        best_est = grid_search_cv.best_estimator_
        print("Best estimator: {0}".format(best_est))
        best_score = grid_search_cv.best_score_
        print("Best score: {0}")
        degrees_scores[(col1, col2), degree] = (best_score, degree, best_est)

print("SUMMARY OF SCORES")
print(degrees_scores)

best_columns = None
best_score = - 100
best_degree = None
best_alpha = None
for cols in degrees_scores:
    r2_aux = degrees_scores[cols]
    if 0 < r2_aux[0] <= 1 and r2_aux[0] > best_score:
        best_columns = cols
        best_score = r2_aux[0]
        best_degree = r2_aux[1]
        best_alpha = r2_aux[2]
print("Highest score = {0}".format(best_score))
print("Columns = {0}".format(best_columns))
print("Best Degree = {0}".format(best_degree))
print("Best Alpha = {0}".format(best_alpha))