# Final Project for CS1430 - EmoVision: A Facial Emotion Recognition Model.

EmoVision is a CNN model trained to classify facial expressions into different emotions. 

## Description
Human emotions are fickle. A smile, while usually associated with the emotion of happiness, can be sad, while tears, usually associated with sadness, can be indicative fear or anger. Most humans, however, are able to distinguish between emotions -- in fact, this ability is fundamental in daily communications and social interactions. Additionally, many emotions and facial expressions are considered universal across different cultures. As such, creating AI systems that can recognize human emotions from expressions would facilitate more seamless interactions between humans and computers, from sociable robots, driver fatigue surveillance, and mental health assessments. 

In this project, we experimented with different CNN architectures to train a model which, given an input image of facial expressions, can classify it into one of seven categories: angry, disgust, fear, happy, sad, surprise, and neutral. The FER-2013 dataset was used. We also tried different data pre-processing techniques, attempting to achieve around the same performance on test set by using just the eyes in the image, rather than the whole face. When this yielded low performance, we generated LIME explainer image to investigate potential reasons why. 

## Final Model Architecture
- The final design we decided to use was based on the VGG16 architecture, which consists of 5 blocks of convolutional layers, each followed by a max pooling layer. 
- The convolutional layers in each block have a filter size of 3x3, and the number of filters increases from 64 to 512 with each successive block. 
- After the VGG16 base layers, there is a feedforward network that includes a dropout layer, two fully connected layers with ReLU activation, and a final output layer with a softmax activation. 
- The model has a total of 13.8 million trainable parameters. During initialization, all the layers of the VGG16 base are frozen to prevent their weights from being updated during training.

## Usage 
- The model can be loaded from different checkpoints by changing the `load_checkpoint` argument in `run.py` to be the path to the weights. 
- The model can be put in evaluation mode by changing `evalute = True`. 
- To generate LIME images, set `explain = True` and change the arguments to `LIME_explainer` to take in the path to the image you want explainer images for. 
- `run.py` also contains a commented line that can be uncommented to run the baseline model. 
- The version that uses just the eyes as input is found in the `eyes` branch in the main repository. 
- To run any of these configurations, make the appropriate edits and run `python3 run.py`. 