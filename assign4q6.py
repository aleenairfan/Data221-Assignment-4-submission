from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = models.Sequential([layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), layers.MaxPooling2D((2,2)), layers.Flatten(), layers.Dense(64, activation='relu'), layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, validation_split=0.1)

print("Test Accuracy:", model.evaluate(X_test, y_test)[1])

# CNNs preserve spatial relationships in images.
# Convolution layers learn features like edges and patterns.