import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

print("4. Building and training the model for different ks")
scores_by_k: dict = {}
best_score_by_k: dict = {}
deep_k: int = 200
acc = np.zeros(deep_k)
std_acc = np.zeros(deep_k)
for k in range(1, deep_k + 1):
    print("FOR K = {0}".format(k))
    knns_mg: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k)
    print("5.1 Getting score with cross val score and scoring accuracy")
    scores: [float] = cross_val_score(knns_mg, df_x_std, df_y, cv=4,
                                      scoring="accuracy")
    print("5.2 Predicting using the model")
    y_pred = cross_val_predict(knns_mg, df_x_std, df_y, cv=4)

    print("Scores = {0}".format(scores))
    scores_by_k[k] = scores
    best_score_by_k[k] = max(scores)
    acc[k - 1] = max(scores)

print("Best k")
best_k = max(best_score_by_k.items(), key=operator.itemgetter(1))[0]
print(best_k)
print("Scores: {0}".format(scores_by_k[best_k]))
print("")

plt.plot(range(1, deep_k+1), acc, 'g')
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
