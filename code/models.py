

"""
Final Project
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TASK 1
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = hp.momentum)

        # TASK 1
        # TODO: Build your own convolutional neural network with a 
        #       15 million parameter budget. The input image will be 
        #       passed through each layer in self.architecture sequentially. 
        #       The imported layers at the top of this file are sufficient
        #       to pass the homework, but feel free to import other layers.
        #
        #       Note 1: 
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Note 2: 
        #       Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note 3: 
        #       Keras layers such as Conv2D and Dense give you the
        #       option of defining an activation function for the layer.
        #       For example, if you wanted ReLU activation on a Conv2D
        #       layer, you'd simply pass the string 'relu' to the
        #       activation parameter when instantiating the layer.
        #       While the choice of what activation functions you use
        #       is up to you, the final layer must use the softmax
        #       activation function so that the output of your network
        #       is a probability distribution.
        #
        #       Note 4: 
        #       Flatten is a useful layer to vectorize activations. 
        #       This saves having to reshape tensors in your network.

        self.architecture = [    

          Conv2D(filters=32, kernel_size = 3, strides = 2, activation = "relu", padding = "same") ,
          Conv2D(filters=32, kernel_size = 3, strides = 2, activation = "relu", padding = "same") ,  
          MaxPool2D(pool_size = 2, strides = 2, padding = "same"),
          Flatten(),
          Dense(100, activation = "relu"),
          Dropout(0.3),
          Dense(15, "softmax")

    # tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(772, activation='relu'),
    # #tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(264, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dense(15, activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """
        # print(tf.shape(x))
        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
       #  print("label",tf.shape(labels))
       #  print("predictions", tf.shape(predictions))
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits = False)
        return loss 

        


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TASK 3
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = hp.momentum)

        # Don't change the below:

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

        # TASK 3
        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained

        for i in range(len(self.vgg16)):
          self.vgg16[i].trainable = False

        # TODO: Write a classification head for our 15-scene classification task.

        self.head = [
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(512, activation="relu", use_bias=True),
               #tf.keras.layers.Dropout(.5),
               tf.keras.layers.Dense(512, activation="relu", use_bias=True),
              #  tf.keras.layers.Dropout(.5),
               tf.keras.layers.Dense(15, activation="softmax", use_bias=True)

        ]

        # Don't change the below:
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

        # TASK 3
        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        #       Read the documentation carefully, some might not work with our 
        #       model!

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

