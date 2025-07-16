import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load image
print("1. Loading the image")
file_name: str = "ladybug-4461314_1280.jpg"
image = imread(file_name)
h, w, _ = image.shape

print("2. Data preprocessing")
print("2.1 Building a vector with all features for each pixel,"
      "R, G, B and relative normalized coordinates")
# Feature vector for all pixels
x_all = np.array([[*image[y, x], x / w, y / h] for y in range(h)
                  for x in range(w)])

print("3. Defining training data, in this case labeled pixels from "
      "leaf, the ladybug shell and a spot in the ladybug picture")
# ---------- Define labeled data ----------
# Training labels (separate from test)
leaf_train = [(100, 100), (300, 100), (800, 100), (600, 1100),
              (500, 300), (450, 1150)]
shell_train = [(400, 640), (410, 630), (420, 645), (390, 650), (430, 640)]
spot_train = [(410, 645), (415, 640), (400, 635), (420, 635), (430, 635)]

print("4. Defining test pixels also belonging to the same areas")
# Test labels (not in training)
leaf_test = [(200, 150), (700, 1200), (580, 900)]
shell_test = [(430, 650), (410, 655), (420, 660)]
spot_test = [(405, 638), (410, 637), (435, 640)]

print("5. Defining x_train, x_test, y_train, y_test:")
print("R,G,B and coordinates x and y from the image are x_train and x_test")
print("Class or area or label is y_train, y_test")
# Format: ((y, x), label)
labeled_train = (
        [(pt, 0) for pt in leaf_train] +
        [(pt, 1) for pt in shell_train] +
        [(pt, 2) for pt in spot_train]
)

labeled_test = (
        [(pt, 0) for pt in leaf_test] +
        [(pt, 1) for pt in shell_test] +
        [(pt, 2) for pt in spot_test]
)

# Training set
x_train = np.array([[*image[y, x], x / w, y / h] for (y, x), _ in
                    labeled_train])
y_train = np.array([label for _, label in labeled_train])

# Test set
x_test = np.array([[*image[y, x], x / w, y / h] for (y, x), _ in labeled_test])
y_test = np.array([label for _, label in labeled_test])

print("6.a Building the random forest")
rf_img: RandomForestClassifier = RandomForestClassifier(n_estimators=100,
                                                        random_state=41)
print("7.a Training the random forest")
rf_start: float = time.time()
rf_img.fit(x_train, y_train)
rf_finish: float = time.time()
# ---------- Accuracy Evaluation ----------
y_pred_rf = rf_img.predict(x_test)
rf_acc_sc: float = accuracy_score(y_test, y_pred_rf)
print("Accuracy score Random Forest = {0}".format(rf_acc_sc))
print("Training time: {0}".format(rf_finish - rf_start))

print("6.b Building the XGBoost")
xgb_img: XGBClassifier = XGBClassifier(n_estimators=100,
                                       random_state=41)
print("7.b Training the XGBoost model")
xgb_start: float = time.time()
xgb_img.fit(x_train, y_train)
xgb_finish: float = time.time()
# ---------- Accuracy Evaluation ----------
y_pred_xgb = xgb_img.predict(x_test)
xgb_acc_sc: float = accuracy_score(y_test, y_pred_xgb)
print("Accuracy score XGBoost = {0}".format(xgb_acc_sc))
print("Training time: {0}".format(xgb_finish - xgb_start))

print("8. Showing segmentations")
# Predict full segmentation
y_pred_all_rf = rf_img.predict(x_all)
rf_segmentation = y_pred_all_rf.reshape((h, w))

y_pred_all_xgb = xgb_img.predict(x_all)
xgb_segmentation = y_pred_all_xgb.reshape((h, w))


# ---------- Visualization ----------
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(rf_segmentation, cmap='tab10')
plt.title(f"Random Forest (Accuracy: {rf_acc_sc: .2f})")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(xgb_segmentation, cmap='tab10')
plt.title(f"XGBoost (Accuracy: {xgb_acc_sc: .2f})")
plt.axis('off')
plt.tight_layout()
plt.show()

