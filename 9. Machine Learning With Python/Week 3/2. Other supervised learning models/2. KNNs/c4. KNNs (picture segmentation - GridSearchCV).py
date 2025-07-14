import operator
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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
print("6. Creating the parameters K for the GridSearchCV model")
k_params: [int] = [x for x in range(1, 10)]
params: dict = {"n_neighbors": k_params}
print("6. Building the KNN Classifier model")
knn_img: KNeighborsClassifier = KNeighborsClassifier()
print("7. Building the GridSearchCV object")
grid_img: GridSearchCV = GridSearchCV(knn_img, param_grid=params, cv=5,
                                      scoring="accuracy")
print("8. Training the model")
grid_img.fit(x_train, y_train)
print("9. Getting best k and best score")
best_k: dict = grid_img.best_params_
best_score: float = grid_img.best_score_
print("Best k = {0}".format(best_k))
print("Best score = {0}".format(best_score))
print("10. Using best K to apply segmentation")
knn_img = KNeighborsClassifier(n_neighbors=best_k["n_neighbors"])
print("10.1 Training the model")
knn_img.fit(x_train, y_train)

print("10.2 Predicting using the model")
# Predict full segmentation
y_pred_all = knn_img.predict(x_all)
segmentation = y_pred_all.reshape((h, w))

# ---------- Visualization ----------
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap='tab10')
plt.title(f"KNN Segmentation (Accuracy: {best_score: .2f})")
plt.axis('off')
plt.show()
