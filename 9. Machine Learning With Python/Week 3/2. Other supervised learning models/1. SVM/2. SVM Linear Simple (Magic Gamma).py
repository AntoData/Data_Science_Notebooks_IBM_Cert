import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

"""
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/
magic/magic04.data

# Data comes from: https://archive.ics.uci.edu/ml/datasets/
MAGIC+Gamma+Telescope

Description:
The MAGIC dataset comes from the MAGIC (Major Atmospheric Gamma Imaging 
Cherenkov) telescope project.

What is MAGIC?
It’s a telescope located in the Canary Islands, specifically designed 
to detect gamma rays.
Gamma rays are extremely energetic forms of light, often emitted by 
cosmic sources like supernovae, pulsars, or black holes.

PROBLEM:
To separate gamma-ray events from hadronic (background) events.

g → Gamma-ray events (signal)
h → Hadronic events (background/noise)

The problem is to classify whether an observed event is likely to be a 
gamma ray or just background noise.

Columns for variable x:
fLength     Major axis of the ellipse
fWidth	    Minor axis of the ellipse
fSize	    Total light content of the shower
fConc	    Concentration of light
fConc1	    Ratio of brightest pixel to total
fAsym	    Asymmetry of the light distribution
fM3Long	    3rd moment along major axis
fM3Trans	3rd moment along minor axis
fAlpha	    Angle of the major axis
fDist	    Distance from image center

Column for variable y:
Original label values:

"g" = gamma event (signal)
"h" = hadron event (background noise)

We need to convert them to 0 and 1
"""

print("1. We need to define the columns names for the dataset")
# Define the column names according to the dataset documentation
column_names = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
    'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
]
print(column_names)
print(" ")

print("2. Read the dataset from the URL")
# Data source
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic" \
      "/magic04.data"

# Read the CSV file with no header, assign column names
df = pd.read_csv(url, header=None, names=column_names)
print(df)

# We filter by class, only classes g and h
df: pd.DataFrame = df[df['class'].isin(['g', 'h'])]
# We transform g to 1 and h to 0
df.loc[:, "class"] = df["class"].apply(
        lambda x: 1 if x == "g" else 0).astype(int)
# We get the independent variable x
df_x: pd.DataFrame = df[['fLength', 'fWidth', 'fSize', 'fConc',
                         'fConc1', 'fAsym', 'fM3Long', 'fM3Trans',
                         'fAlpha', 'fDist']]
# We get the dependent variable y
df_y = df["class"].astype(int)

print("3. Data normalization -> Standard Scaler")
df_x_std: pd.DataFrame = StandardScaler().fit_transform(df_x, df_y)
print("4. Splitting variables x and y into train and test sets")
x_train, x_test, y_train, y_test = train_test_split(df_x_std, df_y)

print("5. Training the model")
lin_svm: LinearSVC = LinearSVC(
    class_weight='balanced', random_state=31, loss="hinge",
    fit_intercept=False, max_iter=100000)
lin_svm.fit(x_train, y_train)

print("6. Using the model to predict values in the test set")
y_pred = lin_svm.decision_function(x_test)

print("7. Using predictions and test sets to get accuracy of the model")
roc_auc_sc: float = roc_auc_score(y_test, y_pred)
print("roc auc sc: {0}".format(roc_auc_sc))
