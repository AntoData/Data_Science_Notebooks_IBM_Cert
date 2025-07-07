import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage.io import imread
from skimage.transform import resize

print("1. We load the image")
url = "ladybug-4461314_1280.jpg"  # high-contrast ladybug on green leaf

image = imread(url)

print("3. We display the image")
h, w, _ = image.shape
plt.imshow(image)
plt.title("Original Ladybug Image (1280x853)")
plt.axis('off')
plt.show()

print("4. We build X as an array which is composed of another array "
      "with the values of the channels R, G and B for each pixel in the "
      "image and their coordinates x and y normalized")
# Create feature vector for each pixel: [R, G, B, x_normalized, y_normalized]
X = []
for y in range(h):
    for x in range(w):
        R, G, B = image[y, x]
        X.append([R, G, B, x / w, y / h])  # normalize coordinates

X = np.array(X)

print("5. We create our training set, labeling random pixels in the image")
# 3️⃣ Define labeled pixels for training (seed points)

# Foreground (ladybug) - label 1
foreground_pixels = [
    (400, 640),  # near center on ladybug shell
    (420, 630),
    (390, 650),
    (410, 660),
    (430, 640),
]

# Background (leaf) - label 0
background_pixels = [
    (100, 100),   # leaf top-left corner
    (50, 1200),   # leaf top-right
    (800, 100),   # leaf bottom-left
    (800, 1200),  # leaf bottom-right
    (500, 300),   # mid left leaf
    (600, 1100),  # mid right leaf
]

labeled_pixels = [(pt, 1) for pt in foreground_pixels] + [(pt, 0) for pt in background_pixels]

# 4️⃣ Prepare training data
X_train = np.array([[*image[y, x], x / w, y / h] for (y, x), _ in labeled_pixels])
y_train = np.array([label for _, label in labeled_pixels])

print("6. We build the KNN model")
knn = KNeighborsClassifier(n_neighbors=3)

print("7. We train the model")
knn.fit(X_train, y_train)

print("8. We use the model and the set x to predict the class of each "
      "pixel in the image")
# Predict every pixel in the image
y_pred = knn.predict(X)

# Reshape flat prediction back into image shape
segmentation = y_pred.reshape((h, w))


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap='gray')
plt.title("Segmented Image (KNN)")
plt.axis('off')

plt.show()