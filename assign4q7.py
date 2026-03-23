from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import models, layers

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = models.Sequential([layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), layers.MaxPooling2D((2,2)), layers.Flatten(), layers.Dense(64, activation='relu'), layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, validation_split=0.1)

pred = model.predict(X_test)
pred_labels = np.argmax(pred, axis=1)

cm = confusion_matrix(y_test, pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

mis = np.where(pred_labels != y_test)[0]

for i in range(3):
    idx = mis[i]
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {pred_labels[idx]}")
    plt.show()

# Errors often occur between visually similar classes.
# Performance can improve with deeper networks or data augmentation.