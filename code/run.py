

"""
Final Project
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import VGGModel, baseline_model
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from tf_explain.core.grad_cam import GradCAM
from tf_explain.callbacks.grad_cam import GradCAMCallback
from keras import Input


from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # First, we create a model that maps the input image to the activations
#     # of the last conv layer as well as the output predictions
#     grad_model = tf.keras.models.Model(
#         [tf.keras.Input(shape=(224, 224, 3), dtype="float32")], [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     # Then, we compute the gradient of the top predicted class for our input image
#     # with respect to the activations of the last conv layer
#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def parse_args():
#     """ Perform command-line argument parsing. """

#     parser = argparse.ArgumentParser(
#         description="Let's train some neural nets!",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '--task',
#         required=True,
#         choices=['1', '3'],
#         help='''Which task of the assignment to run -
#         training from scratch (1), or fine tuning VGG-16 (3).''')
#     parser.add_argument(
#         '--data',
#         default='..'+os.sep+'data'+os.sep,
#         help='Location where the dataset is stored.')
#     parser.add_argument(
#         '--load-vgg',
#         default='vgg16_imagenet.h5',
#         help='''Path to pre-trained VGG-16 file (only applicable to
#         task 3).''')
#     parser.add_argument(
#         '--load-checkpoint',
#         default=None,
#         help='''Path to model checkpoint file (should end with the
#         extension .h5). Checkpoints are automatically saved when you
#         train your model. If you want to continue training from where
#         you left off, this is how you would load your weights.''')
#     parser.add_argument(
#         '--confusion',
#         action='store_true',
#         help='''Log a confusion matrix at the end of each
#         epoch (viewable in Tensorboard). This is turned off
#         by default as it takes a little bit of time to complete.''')
#     parser.add_argument(
#         '--evaluate',
#         action='store_true',
#         help='''Skips training and evaluates on the test set once.
#         You can use this to test an already trained model by loading
#         its checkpoint.''')
#     parser.add_argument(
#         '--lime-image',
#         default='test/Bedroom/image_0003.jpg',
#         help='''Name of an image in the dataset to use for LIME evaluation.''')

#     return parser.parse_args()


def LIME_explainer(model, path, preprocess_fn, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + ".png"
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + ".png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """
     # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, 3, hp.max_num_weights),
        # GradCAMCallback(
        # validation_data=datasets.train_data.__getitem__(1),
        # class_index=0,
        # output_dir='/content/computervisioneers/code/idk',
    # )
    ]
    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        initial_epoch=init_epoch,
         callbacks=callback_list
    )


def test(model, test_data):
    """ Testing routine. """
    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

# def explainer(model, dataset):
#     """Run GradCAM on given image and save the explainer image."""
#     dat = dataset.__getitem__(1)
#     print(np.shape(dat[0][0,:,:,:]))
#     #img = tf.keras.preprocessing.image.load_img(explain_image, target_size=(224, 224))
#     #img = tf.keras.preprocessing.image.img_to_array(img)
#     # Start explainer
#     explainer = GradCAM()
#     model.predict(dat[0][0,:,:,:])
#     grid = explainer.explain((dat[0][0,:,:,:], 3), model, class_index=3) # happy 
#     explainer.save(grid, ".", "grad_cam.png")


def explainer(model, datasets):
  # Prepare image
    # img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # # Make model
    # model = model_builder(weights="imagenet")
    # img = tf.keras.preprocessing.image.load_img(datasets, target_size=(224, 224))
    # img = tf.keras.preprocessing.image.img_to_array(img)
    # print(np.shape(img))
    # print(model.predict(datasets.__getitem__(1)[0]))
    print(model.output)
    print(model.inputs)

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    # preds = model.predict(img_array)
    # print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(datasets.__getitem__(1)[0], model, 'vgg_head')

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()



def main():
    """ Main function. """
    data = '..'+os.sep+'data'+os.sep
    load_vgg = 'vgg16_imagenet.h5'
    load_checkpoint = '/content/computervisioneers/code/checkpoint/vgg.weights.e000-acc0.4094.h5'
    evaluate = False
    explain = True 
    explain_image = '/content/computervisioneers/data/test/happy/im0.png' # the image we want to run gradcam on 

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if load_checkpoint is not None:
        load_checkpoint = os.path.abspath(load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(data):
        data = os.path.abspath(data)
    if os.path.exists(load_vgg):
        load_vgg = os.path.abspath(load_vgg)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(data)

    # if ARGS.task == '1':
    #     model = YourModel()
    #     model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    #     checkpoint_path = "checkpoints" + os.sep + \
    #         "your_model" + os.sep + timestamp + os.sep
    #     logs_path = "logs" + os.sep + "your_model" + \
    #         os.sep + timestamp + os.sep

    #     # Print summary of model
    #     model.summary()
    # else:
    model = VGGModel()
    checkpoint_path = "checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "vgg_model" + \
        os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))

    # Print summaries for both parts of the model
    model.vgg16.summary()
    model.head.summary()

    # Load base of VGG model
    model.vgg16.load_weights(load_vgg, by_name=True)

    # Load checkpoints
    if load_checkpoint is not None:
        print("loading")
        model.head.load_weights(load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if evaluate:
        test(model, datasets.test_data)

        # TODO: change the image path to be the image of your choice by changing
        # the lime-image flag when calling run.py to investigate
        # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg
        # path = ARGS.lime_image
        # LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    elif explain:
        print("explaining")
        LIME_explainer(model, '/content/computervisioneers/data/test/surprised/im1.png', datasets.preprocess_fn, '3')
    else:
        # baseline_model(datasets.test_data) (uncomment for baseline model performance)
        train(model, datasets, checkpoint_path, logs_path, init_epoch)

main()

