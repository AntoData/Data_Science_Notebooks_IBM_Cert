import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

"""
source: https://archive.ics.uci.edu/dataset/267/banknote+authentication

Objective: Classify whether a banknote is genuine or forged based on 
statistical features extracted from the image of the note.

Variable x:
1. variance:	Measures how spread out the pixel values are. 
                Higher variance typically indicates more complex 
                textures.
2. skewness:    Measures the asymmetry of the image data distribution. 
                Skewness can differ between genuine and forged notes.
3. curtosis:    Measures the "tailedness" (extremity) of the 
                distribution. Outliers or sharp edges in the image 
                impact this value.
4. entropy:     Measures the randomness or disorder in the image. 
                Genuine and forged notes often show different entropy 
                patterns.
                
Variable y:
class:      	Class label: 0 = Forged, 1 = Genuine

"""

print("1. We need to define the columns names for the dataset")
# Define the column names according to the dataset documentation
column_names = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "class"
]
print(column_names)
print(" ")

print("2. Read the dataset")
# Read the CSV file with no header, assign column names
df = pd.read_csv("data_banknote_authentication.txt",
                 header=None, names=column_names)
print(df)
# We get the independent variable x
df_x: pd.DataFrame = df[["variance",
                         "skewness",
                         "curtosis",
                         "entropy"]]
# We get the dependent variable y
df_y = df["class"].astype(int)

print("3. Data normalization -> Standard Scaler")
df_x_std: pd.DataFrame = StandardScaler().fit_transform(df_x, df_y)
print("4. We create the SVM model")
lin_svm: LinearSVC = LinearSVC(
    class_weight='balanced', random_state=31, loss="hinge",
    fit_intercept=False, max_iter=100000)

print("5. We build the GridSearchCV")
possible_params: dict = {"C": [10**x for x in range(0, 100000)]}
grid_svm_lin: GridSearchCV = GridSearchCV(lin_svm, param_grid=possible_params,
                                          scoring="accuracy", cv=5)
print("6. Training the model")
grid_svm_lin.fit(df_x_std, df_y)

print("7. Getting best score")
best_scor: float = grid_svm_lin.best_score_
print("Best score = {0}".format(best_scor))

print("8. Getting best parameter C")
best_c: dict = grid_svm_lin.best_params_
print("Best C: {0}".format(best_c))
