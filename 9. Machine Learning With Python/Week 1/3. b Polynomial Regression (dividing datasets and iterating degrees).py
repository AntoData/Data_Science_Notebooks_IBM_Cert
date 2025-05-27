import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
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

# 4. Dividing train and test sets
print("4. Diving our dataset into training and test sets")
x_train, x_test, y_train, y_test = \
    train_test_split(x_scaled, df_life_expectancy[y_data_col])
print("")

# 5. Applying the polynomial transformation
print("5. Applying the polynomial transformation")
degrees = [degree for degree in range(2, 11)]
degree_r2sc: dict = {}

for degree in degrees:
    print("5.1 Transforming our x sets for degree = {0}".format(degree))
    poly_feat_dx = PolynomialFeatures(degree=degree)
    x_train_poly = poly_feat_dx.fit_transform(x_train)
    x_test_poly = poly_feat_dx.fit_transform(x_test)
    print("")

    print("5.2 Training our Linear Model")
    linear_model_poly = LinearRegression()
    linear_model_poly.fit(x_train_poly, y_train)
    print("")

    print("5.3 Using x_test to get our prediction")
    y_pred = linear_model_poly.predict(x_test_poly)
    print("")

    print("5.4 Getting R^2 score")
    r2_sc = r2_score(y_test, y_pred)
    print("R2 = {0}".format(r2_sc))
    degree_r2sc[degree] = r2_sc
    print("")

print("Summary of r2 and degrees")
print(degree_r2sc)
