

"""
Final Project
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
import numpy as np

import hyperparameters as hp
from sklearn.dummy import DummyClassifier

def baseline_model(data):
       '''
       Creates a baseline model using scikit-learn. This model
       only predicts the answer "happy"/class index 3 (the most frequent answer in 
       the training set), ignoring the actual image. 
       '''
       y = np.concatenate([y for x, y in data], axis=0)
       X = np.concatenate([x for x, y in data], axis=0)
       dummy_clf = DummyClassifier(strategy="constant",
                                   constant = tf.tensor([3])) 
       dummy_clf.fit(None, tf.tensor([3])) # fit the model to only predict the index 3
       print(dummy_clf.score(X, y)) # prints the validation accuracy 

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

       # freeze all the layers in VGG 
        for i in range(len(self.vgg16)):
          self.vgg16[i].trainable = False

        # Classification head 
        self.head = [
               Flatten(),
               Dense(units=128, activation="relu"),
               Dropout(0.2),      
               Dense(units=64, activation="relu"),
               Dense(units=15, activation="softmax")]

        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

