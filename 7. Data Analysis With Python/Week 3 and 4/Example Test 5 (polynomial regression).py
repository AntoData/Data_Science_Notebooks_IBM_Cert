"""
EXAMPLE OF QUADRATIC REGRESSION

Source: https://www.mathbits.com/MathBits/TISection/Statistics2/quadratic.html
Dataset is baseball_study.xlsx
"""

# We import first the libraries we are going to need
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

"""
a) Prepare a scatter plot of the data.
"""

# First we load the dataset
baseball_dataset: pd.DataFrame = pd.read_excel("./baseball_study.xlsx")
print(baseball_dataset)

# Now we process the data
# First we rename the columns
baseball_dataset.rename(columns={"Angle\n(degrees)": "Angle",
                                 "Distance\n(feet)": "Distance"}, inplace=True)

# Now we remove the character º from angle so this can be treated as numeric
baseball_dataset["Angle"] = baseball_dataset["Angle"].str.replace("°", "")
print(baseball_dataset)

# Now we set Angle as integer and Distance as float
baseball_dataset["Angle"] = baseball_dataset["Angle"].astype("int")
baseball_dataset["Distance"] = baseball_dataset["Distance"].astype("float")
print(baseball_dataset.dtypes)

# Now let's display a scatter plot where x=Angle and y=distance
plt.scatter(x=baseball_dataset["Angle"], y=baseball_dataset["Distance"])
plt.title("Scatter plot Angle vs Distance")
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance (feet)")
plt.show()

"""
b) Determine a quadratic regression model equation to represent this 
 and graph the new equation.
"""

quadratic_model = np.polyfit(x=baseball_dataset["Angle"],
                             y=baseball_dataset["Distance"],
                             deg=2)
print("Quadratic regression function is:")
print(np.poly1d(quadratic_model))
# For predictions
#print(np.polyval(quadratic_model, 10.0))

"""
c) Decide whether the new equation is a "good fit" to represent this data.
"""
# First we need to generate y_pred values, in order to do so
y_pred = np.polyval(quadratic_model, baseball_dataset["Angle"])
print(y_pred)

mean_squared_value = mean_squared_error(baseball_dataset["Distance"], y_pred)
print("mean_square_error = {0}".format(mean_squared_value))

r2_value = r2_score(baseball_dataset["Distance"], y_pred)
print("r2 = {0}".format(r2_value))

if 0.9 < r2_value <= 1:
    print("It is a good fit")
else:
    print("It is not a good fit")

"""
d) Extrapolate data:  What distance will correspond to an angle of a 5°?
"""
# We apply the function np.polyval to our model with that value
y_pred_5 = np.polyval(quadratic_model, 5)
print("x = 5 ---> y_pred = {0}".format(y_pred_5))

"""
e) Interpolate data:  What angle(s) will correspond to a distance of 
270 feet, to the nearest degree?
"""
# We will perform the same but now x will be distance
quadratic_model_distance = np.polyfit(baseball_dataset["Distance"],
                                      baseball_dataset["Angle"], 2)
print("Formula: {0}".format(quadratic_model_distance))
# We will predict the value now
y_pred_270_feet = np.polyval(quadratic_model_distance, 270)
print("If distance is 270 feet, angle is {0}º".format(y_pred_270_feet))

"""
f) The first baseman is positioned 100 feet from home plate and the 
right fielder is positioned 180 feet from home plate. The batter wants 
to hit the ball half way between these players. What angle, to the 
nearest degree, should be used to accomplish this hit? Round to 3 
decimal places.
"""
# Let's find what distance is halfway
distance_half = (180 + 100) / 2
# Now we will predict the value
y_pred_distance_half = np.polyval(quadratic_model_distance, distance_half)
print("If distance is {0} feet, angle is {1}º".format(distance_half,
                                                      y_pred_distance_half))