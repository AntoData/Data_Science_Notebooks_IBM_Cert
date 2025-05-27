import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score

"""
SOURCE:
USGS “All Earthquakes, Past Month”
https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv

Independent variable X columns:
Variable	Physical Rationale
depth (km)	Seismic attenuation & energy reaching surface. 
            Deeper quakes lose more energy before reaching the surface, 
            so depth modulates felt intensity and thus the chance of 
            “significant” classification.
latitude, 
longitude	Tectonic/regional effects. Different fault systems and 
            plate‐boundary geometries produce different typical 
            magnitudes. Spatial coordinates serve as proxies for 
            those geologic settings.
gap (°)	    Data‐quality / detection geometry. “Gap” is the largest 
            azimuthal gap in station coverage around the event. 
            Large gaps often mean poorer‐constrained hypocenters—and 
            tend to occur for smaller or more remote events—introducing
             bias in who gets detected as “major.”

Dependent variable y:
mag:    Classified in the following 
        Minor - [0.0, 3.0) 
        Light - [3.0, 5.0)
        Moderate - [5.0, 7.0)
        Strong - [7.0, 10.0]
"""
# 1. We open the dataset
df: pd.DataFrame = pd.read_csv('USGS_all_earthquakes_past_month.csv')
# 2. Preprocessing data
# 2.1 Dropping rows with NA values
df.dropna(inplace=True, axis=0)
print(df)
# 2.2 Getting independent variable x
df_x = df[['depth','latitude', 'longitude','gap']]
# 2.3 Diving column mag in the classes described above for variable y
min_val = df['mag'].min() - 1e-6
max_val = df['mag'].max() + 1e-6
bins = [min_val, 3.0, 5.0, max_val]
df_y = pd.cut(df["mag"], labels=["Minor", "Light", "Moderate", "Strong"],
              bins=[min_val, 3.0, 5.0, 7.0, max_val])
print(df_y)

# 3. Applying Standard Scaler
std_scl: StandardScaler = StandardScaler()
std_scl.fit(df_x, df_y)
df_scl: pd.DataFrame = std_scl.transform(df_x)

# 4. Diving the dataset in training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_scl, df_y,
                                                    test_size=0.3)

"""
STRATEGY:
1. One vs All
Binary classifier for each class.
Binary classifier one vs all the other classes
"""
# 5.1 We build the logistic regression model
log_model: LogisticRegression = LogisticRegression()
# 6.1 We build the one vs all classifier and pass the logistic model
# object as an input
ova_model: OneVsRestClassifier = OneVsRestClassifier(log_model)
# 7.1 We train the model
log_model.fit(x_train, y_train)
# 8.1 We use x testing set to predict values to be able to check
# the accuracy of the model
y_pred: pd.DataFrame = log_model.predict(x_test)
# 9.1 Getting the accuracy score of the model
acc_scr = accuracy_score(y_test, y_pred)
print("One vs All: Accuracy score = {0}".format(acc_scr))

"""
2. One vs One
Trains a binary classifier for every pair of classes
During prediction all classifiers are used and a voting mechanism
decides the final class
"""
# 5.2 We create the logistic model object
log_model_2: LogisticRegression = LogisticRegression()
# 6.2 We create the strategy One vs One classifier object and use the
# logistic model object as input
ovo_model: OneVsOneClassifier = OneVsOneClassifier(log_model_2)
# 7.2 We train the model
ovo_model.fit(x_train, y_train)
# 8.2 We use the testing set of variable x to predict values so we
# can use it to get the accuracy of the model
y_pred: pd.DataFrame = ovo_model.predict(x_test)
# 9.2 We get the accuracy score
acc_scr: float = accuracy_score(y_pred, y_test)
print("One vs One: Accuracy score = {0}".format(acc_scr))

"""
3. Multinomial default
"""
# 5.3 We build the logistic model
log_mn: LogisticRegression = LogisticRegression()

# 6.3 We don't need an additional object for this strategy
# 7.3 We train the logistic model
log_mn.fit(x_train, y_train)

# 8.3 We use training set for variable x to make predictions we can
# use to get the accuracy of the model
y_pred: pd.DataFrame = log_mn.predict(x_test)

# 9.3 We get the accuracy score now
acc_scr: float = accuracy_score(y_pred, y_test)
print("Multinomial: Accuracy score = {0}".format(acc_scr))