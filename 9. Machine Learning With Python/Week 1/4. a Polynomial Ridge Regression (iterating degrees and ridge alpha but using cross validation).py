import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge

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
alpha_values: [int] = [0.0001, 0.001, 0.01, 0.1, 1, 10]

for degree in degrees:
    print("4.1 Applying the polynomial transformation for degree = {0}".format(
        degree))
    poly_feat_dx = PolynomialFeatures(degree=degree)
    x_poly_scaled = poly_feat_dx.fit_transform(x_scaled)
    print("")

    for alpha_v in alpha_values:
        print("4.2 Iterating through alpha={0} values for Ridge Model".format(
            alpha_v))
        ridge_model: Ridge = Ridge(alpha=alpha_v)
        print("")

        print("4.3 Using cross validation with our Ridge model to get scores")
        scores_d_a = cross_val_score(ridge_model, x_poly_scaled,
                                     df_life_expectancy[y_data_col], cv=5)
        print("scores = {0}".format(scores_d_a))
        mean_score_d_a = scores_d_a.mean()
        print("mean score = {0}".format(mean_score_d_a))
        std_score_d_a = scores_d_a.std()
        print("STD of scores = {0}".format(std_score_d_a))
        print("")

        print("4.4 Predicting values using cross validation")
        y_pred = cross_val_predict(ridge_model, x_poly_scaled,
                                   df_life_expectancy[y_data_col], cv=5)
        print(y_pred)
        print("")

        degrees_scores[(degree, alpha_v)] = mean_score_d_a, std_score_d_a

print("SUMMARY OF SCORES")
print(degrees_scores)

r2_max = 0
r2_std = 0
key_max = None
for k_ in degrees_scores:
    r2_aux = degrees_scores[k_]
    if 0 < r2_aux[0] <= 1 and r2_aux[0] > r2_max:
        key_max = k_
        r2_max = r2_aux[0]
        r2_std = r2_aux[1]
print("Highest score = {0}".format(r2_max))
print("STD of score in different sets = {0}".format(r2_std))
print("Degree and alpha = {0}".format(key_max))
