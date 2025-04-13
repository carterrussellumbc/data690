import pathlib
import numpy as np
import tensorflow as tf
from helpers import Helpers
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class ImageClassifier(Helpers):


    def __init__(self):
        self.batch_size = 128
        self.img_height = 96
        self.img_width = 96
        self.helpers = Helpers()

    def train_model(self, val_split, seed, img_dir, class_num, epoch_num, model_name):
        img_dir = pathlib.Path(img_dir)
        train_ds = tf.keras.utils.image_dataset_from_directory(
          img_dir,
          validation_split=val_split,
          subset="training",
          seed=seed,
          image_size=(self.img_height, self.img_width),
          batch_size=self.batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
          img_dir,
          validation_split=val_split,
          subset="validation",
          seed=seed,
          image_size=(self.img_height, self.img_width),
          batch_size=self.batch_size)

        num_classes = class_num

        model = tf.keras.Sequential([
          # tf.keras.layers.Rescaling(1./255),
          # tf.keras.layers.RandomFlip("horizontal_and_vertical", 96024),
          # tf.keras.layers.RandomBrightness([-0.2,0.2], seed=96024),
          # tf.keras.layers.RandomRotation(0.2, fill_mode='reflect', interpolation='bilinear', seed=96024),
          tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(num_classes)
        ])

        tf.keras.optimizers.Adam.learning_rate = 0.0001
        tf.keras.optimizers.Adamax.learning_rate = 0.0001

        model.compile(
          optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

        model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epoch_num
        )

        self.helpers.generate_roc_curve_data(model, val_ds, train_ds)

        # model.save(model_name)
