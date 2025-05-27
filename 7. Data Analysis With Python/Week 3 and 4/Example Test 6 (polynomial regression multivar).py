import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

"""
In order to use multivariable polynomial regression we have to
a) Create training datasets for x and y
b) We have to create a PolynomialFeatures object with the corresponding
degree and then use fit_transform in our training dataset X, getting
a new polynomial transformed array X_poly
c) We need to create a LinearRegression object and use fit with our new
polynomial transformed array X and our training dataset Y
d) We need to transform the data in X we want to use to predict Y to
a polynomial array using our polynomial features object´s function
transform, as a result we get the input for predictions
e) Now, we use predict in our LinearRegression model using the 
transformed array we got in the step above as input and we get our 
prediction
"""

"""
a) Create training datasets for x and y
"""
# Load the California housing dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

# Select appropriate features from the dataset
# Assuming you want to use features like 'MedInc' (median income),
# 'HouseAge', and 'AveRooms' (average rooms)
X = df[['MedInc', 'HouseAge', 'AveRooms']].values
y = df['PRICE'].values
"""
b) We have to create a PolynomialFeatures object with the corresponding
degree and then use fit_transform in our training dataset X, getting
a new polynomial transformed array X_poly
"""
# Polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(X_poly)

"""
c) We need to create a LinearRegression object and use fit with our new
polynomial transformed array X and our training dataset Y
"""
# Linear Regression model
model = LinearRegression()
model.fit(X_poly, y)


"""
d) We need to transform the data in X we want to use to predict Y to
a polynomial array using our polynomial features object´s function
transform, as a result we get the input for predictions
"""
# New data with median income, house age, and average number of rooms
new_data = np.array([[3, 20, 5]])  # Example values for median income,
# house age, and average rooms
new_data_poly = poly.transform(new_data)

"""
e) Now, we use predict in our LinearRegression model using the transformed
 array we got in the step above as input and we get our prediction
"""
# Predicting the price
predicted_price = model.predict(new_data_poly)
print(predicted_price)
