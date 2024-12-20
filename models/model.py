import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Normalize images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save('mnist_model.h5')
