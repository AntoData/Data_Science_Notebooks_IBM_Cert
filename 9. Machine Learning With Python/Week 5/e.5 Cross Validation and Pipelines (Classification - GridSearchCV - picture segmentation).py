import numpy as np
import pandas as pd
import seaborn as sns
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

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

print("3. Creating the pipeline")
pipeline_: Pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier_knn', KNeighborsClassifier())
])

print("3.1 Pipeline parameters")
param_grid_: {str: [int]} = {'pca__n_components': [3, 4, 5],
                             'classifier_knn__n_neighbors':
                                 [i for i in range(2, 10)]
                             }
print(param_grid_)

print("4. We need to use StratifiedKFold in order to use it later")
cv_: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=42)


print("Creating now our GridSearchCV to optimise the model")
grid_search_cv: GridSearchCV = GridSearchCV(
    estimator=pipeline_,
    param_grid=param_grid_,
    cv=cv_,
    scoring="accuracy",
    verbose=2
)

print("5. Training the model")
grid_search_cv.fit(x_train, y_train)

print("6. Best model parameters are: ")
print(grid_search_cv.best_params_)

print("7. Getting model's score")
test_score: float = grid_search_cv.score(x_test, y_test)
print("Model's score = {0}".format(test_score))

print("8. Using the model to predict x_test")
y_pred: pd.DataFrame = grid_search_cv.predict(x_test)

print("9. Confusion Matrix")
classes_labels: {int: str} = {
    k: v for k, v in zip(range(0, len(np.unique(y_test))), np.unique(y_test))}
confusion_matrix_knn: np.ndarray = confusion_matrix(y_test, y_pred)
print(confusion_matrix_knn)
confusion_matrix_svm_knn = \
    confusion_matrix_knn.astype(int).astype(str)
fig, axes = plt.subplots(1, 1, figsize=(12, 5))
sns.heatmap(confusion_matrix_knn, annot=True, cmap='Blues', fmt='d',
            ax=axes, xticklabels=list(classes_labels.values()),
            yticklabels=list(classes_labels.values()),
            annot_kws={'color': 'black'}, )
axes.set_title('KNN Testing Confusion Matrix')
axes.set_xlabel('Predicted')
axes.set_ylabel('Actual')
plt.tight_layout()
plt.show()

print("10. Getting precision by class")
precision_scores_knn: np.ndarray = \
    precision_score(y_test, y_pred, average=None)
print("10.1 Going class by class")
for i in range(0, len(precision_scores_knn)):
    print("Class = {0}".format(classes_labels[i]))
    print("KNN = {0}".format(precision_scores_knn[i]))
    print("Ratio of instances classified as {0} "
          "that truly belong to class {0}".format(classes_labels[i]))
    print("")

print("11. Getting Recall scores by class")
recall_score_knn: np.ndarray = recall_score(y_test, y_pred, average=None)
for i in range(0, len(recall_score_knn)):
    print("For class = {0}".format(classes_labels[i]))
    print("Recall score for KNN = {0}".format(
        recall_score_knn[i]))
    print("How many instances of class {0} there were and how many we "
          "found".format(classes_labels[i]))
    print("")

print("7. Showing best segmentation")
# Predict full segmentation
y_pred_all = grid_search_cv.predict(x_all)
segmentation = y_pred_all.reshape((h, w))

# ---------- Visualization ----------
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap='tab10')
plt.title("KNN Segmentation (Accuracy: {0})".format(
    grid_search_cv.best_score_))
plt.axis('off')
plt.show()
