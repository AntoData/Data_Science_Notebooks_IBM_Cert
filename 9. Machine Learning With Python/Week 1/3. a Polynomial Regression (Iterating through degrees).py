import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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

print("4. We will transform into a polynomial linear regression")
degrees: [int] = [x for x in range(2, 11)]
r2_scores: dict = {}
print("Degrees we will check: {0}".format(degrees))
for degree in degrees:
    print("4. 1 - Trying degree = {0}".format(degree))
    poly_dx = PolynomialFeatures(degree=degree)
    x_poly_scaled = poly_dx.fit_transform(x_scaled)

    print("4. 2 - Training Linear Model")
    linear_model = LinearRegression()
    linear_model.fit(x_poly_scaled, df_life_expectancy[y_data_col])

    print("4. 3 - Getting prediction")
    y_pred = linear_model.predict(x_poly_scaled)

    print("4. 4 - Getting R^2 score")
    r2_sc = r2_score(df_life_expectancy[y_data_col], y_pred)
    print("R^2 score = {0}".format(r2_sc))
    r2_scores[degree] = r2_sc

print(r2_scores)
