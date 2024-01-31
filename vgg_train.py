# https://dataman-ai.medium.com/transfer-learning-for-image-classification-5-get-image-data-ready-and-go-554044a12e6d
# https://dataman-ai.medium.com/transfer-learning-for-image-classification-6-build-the-transfer-learning-model-67d87999af4a
# https://github.com/dataman-git/codes_for_articles/blob/master/20220804_Transfer_learning_for_Image_Classification.ipynb

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications.imagenet_utils import decode_predictions

IMAGES_FOLDER = os.path.join('.', 'images')
LABELS = [0, 1]
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 15

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    IMAGES_FOLDER,
    # labels=LABELS,
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    IMAGES_FOLDER,
    # labels=LABELS,
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

VGG16_features=tf.keras.applications.VGG16(input_shape=IMAGE_SHAPE,
                                           include_top=False,
                                           weights='imagenet')
VGG16_features.trainable=False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')
fc1 = tf.keras.layers.Dense(64, activation='relu')

model = tf.keras.Sequential([
  VGG16_features,
  global_average_layer,
  fc1,
  prediction_layer
])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              # loss=tf.keras.losses.binary_crossentropy,
              loss=tf.keras.losses.sparse_categorical_crossentropy, # Calculates how often predictions match integer labels.
              metrics=["accuracy"])

steps_per_epoch = len(train_ds)
validation_steps = len(test_ds)

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    validation_data=test_ds)

model.save(os.path.join('.', 'trained_categorical_vgg'))

loss0,accuracy0 = model.evaluate(test_ds, steps = validation_steps)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))