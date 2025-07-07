import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score

"""
SOURCE: 
#   VizieR Astronomical Server vizier.cds.unistra.fr
#    Date: 2025-05-12T15:26:56 [V7.4.6]
#   In case of problem, please report to:	cds-question@unistra.fr

Independent variable x:
B-V	Color:
index (difference between blue and visual magnitude)	
Strongest predictor of stellar temperature and thus spectral class.

Vmag:
Apparent visual magnitude	
Combined with parallax, it gives absolute magnitude (luminosity), 
which helps distinguish giants from dwarfs.

Plx:
Parallax (in milliarcseconds)
Used to estimate distance → essential for luminosity calculation.

Dependent variable y:
SpType:
Spectral type (e.g., “F5”, “K3V”, “G8III”)
This is the target of classification.
For modeling, you can extract just the spectral class letter ('F', 'K', 
etc.) as the label.
Astrophysical basis: Spectral type is a function of the star’s 
effective temperature, metallicity, and surface gravity, which 
correlate with observables like color index and luminosity.
"""
# 1. We open the dataset
df_stars: pd.DataFrame = pd.read_csv('vizier_stars_sampled_10000.tsv',
                                     delimiter="\t")
print(df_stars)
print(df_stars.dtypes)

# 2. Preprocessing
# 2.1 Drop NA
df_stars.dropna(inplace=True)
# 2.2 Sample 1000 records
df_stars = df_stars.sample(n=1000, random_state=42)
# 2.3 Getting only the first letter in the star type
df_stars['SpTypeSimple'] = df_stars['SpType'].str[0]
# No need for more preprocessing

# 3. We compose variable x and variable y
df_stars_x: pd.DataFrame = df_stars[['Vmag', 'Plx', 'B-V']]
df_stars_y: pd.DataFrame = df_stars['SpTypeSimple']

# 4. We apply the Standard Scaler
std_scl: StandardScaler = StandardScaler()
std_scl.fit(df_stars_x)
df_stars_x_std_scl: pd.DataFrame = std_scl.transform(df_stars_x)

# 5. We split the dataset between training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_stars_x_std_scl,
                                                    df_stars_y)

"""
Now we will apply the different strategies to solve this multiple 
regression problem
"""

"""
1. One Vs One
Trains a binary classifier for every pair of classes
During prediction all classifiers are used and a voting mechanism
decides the final class
"""
# 6.1 We create the logistic model that will be the input for the One
# vs One strategy
log_reg: LogisticRegression = LogisticRegression(max_iter=200)

# 7.1 We create now the One Vs One classifier based on Logistic Reg
model_log_ovo: OneVsOneClassifier = OneVsOneClassifier(log_reg)

# 8.1 We train the model using the training set for variable x
model_log_ovo.fit(x_train, y_train)

# 9.1 We use the test set to predict values
y_pred: pd.DataFrame = model_log_ovo.predict(x_test)

# 10.1 We use now y_test and y_pred to get the accuracy score of this
# method
acc_scr: float = accuracy_score(y_test, y_pred)
print("One vs One -> Accuracy score = {0}".format(acc_scr))


"""
2. One Vs All
Binary classifier for each class.
Binary classifier one vs all the other classes
"""
# 6.1 We create the logistic model that will be the input for the One
# vs All strategy
log_model: LogisticRegression = LogisticRegression()

# 7.2 We create now the One Vs All classifier based on Logistic Reg
ova_model: OneVsRestClassifier = OneVsRestClassifier(log_reg)

# 8.2 We train the model using x_train and y_train
ova_model.fit(x_train, y_train)

# 9.2 We use x_test to get predictions to get the accuracy level of the
# model
y_pred: pd.DataFrame = ova_model.predict(x_test)

# 10.2 We work out the accurary score of this model
acc_scr: float = accuracy_score(y_test, y_pred)
print("One Vs All -> Accuracy score = {0}".format(acc_scr))

"""
3. Multinomial
"""
# 6.1 We create the Logistic Model, by default now the strategy applied
# will be multinomial
log_model_multi: LogisticRegression = LogisticRegression()

# 7.1 We train the model using x_train, y_train
log_model_multi.fit(x_train, y_train)

# 8.1 We use x_test to make predictions later used for accuracy score
y_pred = log_model_multi.predict(x_test)

# 9.1 Now we use y_test and y_test to get accuracy score
acc_scr: float = accuracy_score(y_test, y_pred)
print("Multinomial -> Accuracy score = {0}".format(acc_scr))