# This code has been used on Google Colaboratory service for model training
# but can easily be used elsewhere with change to downloading and savin the model

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import seaborn
from tensorflow.keras import datasets, layers, models, metrics
from tensorflow.keras.optimizers import SGD
from IPython.core import history
from sklearn.metrics import confusion_matrix
from google.colab import files

def augmentImages(images):
  augmentedImages = []
  for img in images:
    # ROTATION
    angle = random.randint(-45, 45)
    # print('Angle: ' + str(angle) + '°')
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # ZOOM
    (h, w) = img.shape[:2]
    zoomFactor = random.uniform(.6, 1.25)
    # print("Zoom: " + "{:.2f}".format(zoomFactor))
    new_h, new_w = int(h * zoomFactor), int(w * zoomFactor)

    # Change image size - zoom in if factor > 1, zoom out if factor < 1
    zoomed_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if zoomFactor > 1:
      start_row, start_col = int((zoomed_image.shape[0] - h) / 2), int((zoomed_image.shape[1] - w) / 2)
      img = zoomed_image[start_row:start_row+h, start_col:start_col+w]
    elif zoomFactor < 1:
      pad_top = (h - zoomed_image.shape[0]) // 2
      pad_bottom = h - pad_top - zoomed_image.shape[0]
      pad_left = (w - zoomed_image.shape[1]) // 2
      pad_right = w - pad_left - zoomed_image.shape[1]

      img = cv2.copyMakeBorder(zoomed_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    # NOISE
    noiseLevel = random.randint(10, 50)
    # print("Nois level: " + str(noiseLevel))
    for i in range(noiseLevel):
      color = random.randint(0,180)
      row = random.randint(0,27)
      column = random.randint(0,27)
      img[row, column] = color

    augmentedImages.append(img)

  return np.array(augmentedImages)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = augmentImages(train_images)
test_images = augmentImages(test_images)

train_images.shape

train_images.max()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images.max()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

epochs = 20
batch_size = 64
num_classes = 10

train_label_cat = tf.keras.utils.to_categorical(train_labels, num_classes)
test_label_cat = tf.keras.utils.to_categorical(test_labels, num_classes)

opt = SGD(learning_rate=0.001, momentum=0.9)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics.CategoricalAccuracy())

history = model.fit(train_images, train_label_cat, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_label_cat))

plt.plot(history.history['categorical_accuracy'], label='accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_label_cat, verbose=2)

model_path = "models/model.h5"
model.save(model_path)

# files.download(model_path)