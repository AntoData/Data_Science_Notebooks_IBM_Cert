import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score

"""
Source: https://data.nasa.gov/dataset/meteorite-landings

Independent variable x
Columns:
GeoLocation: Divided in
Latitude: Geographical coordinates of the meteorite landing. These can 
help identify regional patterns in meteorite distribution.
Longitude: Same as above

Year: The year the meteorite was found or observed falling. This can 
capture temporal trends in meteorite discoveries.

Fall Status: Indicates whether the meteorite was observed falling 
("Fell") or was found later ("Found"). This categorical variable can 
be encoded numerically for modeling.

Recclass: The classification of the meteorite, indicating its 
composition. This categorical variable can be transformed using 
one-hot encoding to be used in the model.
"""
# 1. Open the CSV file
df: pd.DataFrame = pd.read_csv('Meteorite_Landings.csv')

# 2. Data preprocessing
# 2.1 Drop rows with empty values
df.dropna(inplace=True)

# 2.2 Transform the column fall_flag to numeric
df['fall_flag'] = [0 if x == "Fell" else 1 for x in df['fall']]

# 2.3 Separate the coordinates in Geolocation into two numeric fields
# 2.3.a First, clean the field, remove the () and split by ,
df['GeoLocation_clean'] = \
    df['GeoLocation'].str.replace("(", "").str.replace(")", "").str.split(",")
# 2.3.b Second create a new column for each value in the array created
# in each row above
df["Latitude"] = [x[0] for x in df["GeoLocation_clean"]]
df["Longitude"] = [x[1] for x in df["GeoLocation_clean"]]

# 2.4 We have to convert the column recclass that has multiple possible
# values into numeric. In order to do so, we will create a new
# column for each value and assign 0 if that row did not have that
# value or 1 if it did
recclass_encoded = pd.get_dummies(df['recclass'], prefix='class')
df = pd.concat([df, recclass_encoded], axis=1)

# 2.5 Convert mass_class to 3 different groups
mass_min: float = df['mass (g)'].min() - 1
mass_max: float = df['mass (g)'].max() + 1
df['mass_class']: pd.DataFrame = pd.cut(df['mass (g)'],
                                        bins=[mass_min, 1000, 10000,
                                              mass_max],
                                        labels=["Small", "Medium", "Large"])

# 2.6 Leave only the columns we need in the dataframe
df_clean: pd.DataFrame = df.drop(columns=["name", "id", "recclass",
                                          "reclat", "reclong", "GeoLocation",
                                          "mass (g)", "fall", "nametype",
                                          "GeoLocation_clean"])

# 2.7 Create independent variable x with the corresponding columns
df_x: pd.DataFrame = df_clean.drop(columns=["mass_class"])
print(df_x.columns)
# 2.8 Create dependent variable y with the corresponding column
df_y = df_clean["mass_class"]

# 3. Apply the standard scaler to variable x
std_scl: StandardScaler = StandardScaler()
std_scl.fit(df_x, df_y)
df_x_scl: pd.DataFrame = std_scl.transform(df_x)

# 4. Divide the variables into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(df_x_scl, df_y)

"""
STRATEGY
1. One vs One
Trains a binary classifier for every pair of classes
During prediction all classifiers are used and a voting mechanism
decides the final class
"""
# 5.1 We create the object for logistic regression, we need to apply the
# solver lbfgs and set a max number of iterations
log_model: LogisticRegression = LogisticRegression(solver='lbfgs',
                                                   max_iter=1000)

# 6.1 We create the object for the One vs One classification strategy
# as pass the logistic model object as input
ovo_model: OneVsOneClassifier = OneVsOneClassifier(log_model)

# 7.1 We train the One vs One logistic model using training sets
ovo_model.fit(x_train, y_train)

# 8.1 We use the test set of variable x to make some predictions
y_pred: pd.DataFrame = ovo_model.predict(x_test)

# 9.2 We use those predictions and the test set of variable y to get
# the accuracy score of the model and see how well it does
acc_scr: float = accuracy_score(y_test, y_pred)
print("One vs One: Accuracy score = {0}".format(acc_scr))

"""
2. One vs All
Binary classifier for each class.
Binary classifier one vs all the other classes
"""
# 5.2 We create the object for the logistic regression model
log_model_2: LogisticRegression = LogisticRegression()

# 6.2 We create the object for the One vs All classification strategy
# and pass the logistic model object as input
ova_model: OneVsRestClassifier = OneVsRestClassifier(log_model_2)

# 7.2 We train our One vs All model using our training sets
ova_model.fit(x_train, y_train)

# 8.2 We use the testing set of variable x to make some prediction
y_pred_2: pd.DataFrame = ova_model.predict(x_test)

# 9.2 We use these predictions and the testing set of variable y to
# get the accuracy score
acc_scr_2: float = accuracy_score(y_test, y_pred_2)
print("One vs All: Accuracy score = {0}".format(acc_scr_2))

"""
3. Multinomial default
"""
# 5.3 We create the logistic regression model object
log_model_3: LogisticRegression = LogisticRegression()

# 6.3 We train the model using our training sets
log_model_3.fit(x_train, y_train)

# 7.3 We use our testing set for variable x to make some predictions
y_pred_3: pd.DataFrame = log_model_3.predict(x_test)

# 8.3 We use these predictions and the testing set of variable y to get
# the accuracy score
acc_scr_3: float = accuracy_score(y_test, y_pred_3)
print("Multinomial: Accuracy score = {0}".format(acc_scr_3))
