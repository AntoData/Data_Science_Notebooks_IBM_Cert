import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""
SOURCE: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
"""
# 1. First we open the file and explore its data
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
print("2.2 Analysing the correlation data")
# First we remove the row that correlates Life expectancy with itself
corr_life_expectancy: pd.DataFrame = corr_life_expectancy.iloc[2:]
print("2.3 Getting highest correlation relationship")
highest_correlation: pd.Series = corr_life_expectancy.idxmax()
print(highest_correlation)
print("2.4 Getting second highest correlation relationship")
corr_life_expectancy = corr_life_expectancy.drop(
    index=[highest_correlation.iloc[0]])
highest_correlation2: pd.Series = corr_life_expectancy.idxmax()
print(highest_correlation2)
x_1_col: str = highest_correlation.iloc[0]
x_2_col: str = highest_correlation2.iloc[0]
print("So our independent variables are:")
print("- {0}".format(x_1_col))
print("- {0}".format(x_2_col))
print("")

# 3. Applying StandardScaler
print("3. Applying Standard Scaler to x")
std_scaler = StandardScaler()
std_scaler.fit(df_life_expectancy[[x_1_col, x_2_col]],
               df_life_expectancy[y_data_col])
x_scaled = std_scaler.transform(df_life_expectancy[[x_1_col, x_2_col]])
print("")

# 4. Applying Polynomial Transformation
print("4. Applying the polynomial transformation")
degrees = [degree for degree in range(2, 11)]
print("Degrees = {0}".format(degrees))
degrees_scores: dict = {}

for degree in degrees:
    print("4.1 Applying the polynomial transformation for degree = {0}".format(
        degree))
    poly_feat_dx = PolynomialFeatures(degree=degree)
    x_poly_scaled = poly_feat_dx.fit_transform(x_scaled)
    print("")

    print("4.2 Creating the Linear Model")
    linear_model = LinearRegression()
    print("")

    print("4.3 Applying cross validation to get r2 scores")
    r2_scores_cv = cross_val_score(linear_model, x_poly_scaled,
                                   df_life_expectancy[y_data_col], cv=5)
    print("Scores = {0}".format(r2_scores_cv))
    r2_scores_cv_mean = r2_scores_cv.mean()
    print("Mean scores = {0}".format(r2_scores_cv_mean))
    print("")

    print("4.4 Applying cross validation to get predictions")
    predictions = cross_val_predict(linear_model, x_poly_scaled,
                                    df_life_expectancy[y_data_col], cv=5)
    print("")

    print("4.5 Using predictions to get an alternative measure to R2 scores")
    r2_scores_pred = r2_score(df_life_expectancy[y_data_col], predictions)
    print("Mean scores through predictions = {0}".format(r2_scores_pred))
    degrees_scores[degree] = [r2_scores_cv, r2_scores_cv_mean,
                              r2_scores_pred]
    print("")

print("SUMMARY OF SCORES")
print(degrees_scores)
