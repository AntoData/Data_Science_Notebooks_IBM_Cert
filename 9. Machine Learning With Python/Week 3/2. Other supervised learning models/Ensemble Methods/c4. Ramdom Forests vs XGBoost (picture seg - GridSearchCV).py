import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import GridSearchCV
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
rf_img: RandomForestClassifier = RandomForestClassifier(random_state=41)
print("7.a Creating the GridSearchCV Object")
rf_param_grid: dict = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf: GridSearchCV = GridSearchCV(rf_img, param_grid=rf_param_grid,
                                            cv=5, scoring="accuracy")
print("8.a Training the model")
rf_start: float = time.time()
grid_search_rf.fit(x_train, y_train)
rf_finish: float = time.time()
print("Training time = {0}".format(rf_finish-rf_start))
print("Best score = {0}".format(grid_search_rf.best_score_))
print("Best params = {0}".format(grid_search_rf.best_params_))

print("6.b Building the XGBoost")
xgb_img: XGBClassifier = XGBClassifier(random_state=41)
print("7.b Creating the GridSearchCV object")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search_xgb: GridSearchCV = GridSearchCV(xgb_img,
                                             param_grid=xgb_param_grid,
                                             cv=5, scoring="accuracy")
print("7.b Training the model")
xgb_start: float = time.time()
grid_search_xgb.fit(x_train, y_train)
xgb_finish: float = time.time()
print("Training time = {0}".format(xgb_finish-xgb_start))
print("Best score = {0}".format(grid_search_xgb.best_score_))
print("Best params = {0}".format(grid_search_xgb.best_params_))

print("8. Showing segmentations")
# Predict full segmentation
rf_img_best: RandomForestClassifier = grid_search_rf.best_estimator_
y_pred_all_rf = rf_img_best.predict(x_all)
rf_segmentation = y_pred_all_rf.reshape((h, w))

xgb_img_best: XGBClassifier = grid_search_xgb.best_estimator_
y_pred_all_xgb = xgb_img_best.predict(x_all)
xgb_segmentation = y_pred_all_xgb.reshape((h, w))


# ---------- Visualization ----------
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(rf_segmentation, cmap='tab10')
plt.title(f"Random Forest (Accuracy: {grid_search_rf.best_score_: .2f})")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(xgb_segmentation, cmap='tab10')
plt.title(f"XGBoost (Accuracy: {grid_search_xgb.best_score_: .2f})")
plt.axis('off')
plt.tight_layout()
plt.show()

