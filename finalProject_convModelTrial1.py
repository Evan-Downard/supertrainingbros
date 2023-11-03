"""
Gabe Koeller

This code will be used or referenced in the final product of the CS 4331 Deep Learning final project
for the group "Super Training Bros."

This code will produce a convolutional neural network using TensorFlow to use in a reinforcement learning
model to determine optimal play in the atari version of "Mario Bros." as modeled in the Gymnasium environment.
The action space specifies 18 possible actions (no action, cardinal directional input, fire, and all combinations
of the previously listed actions) and as such, the model must output in a onehot format to select only one action.
The game environment is given as a multidimensional array with size 210 x 160. The game can then be in rgb or
grayscale format. While this has little effect on the code of the model, it does need to be taken into account
when determining how complex the model should be since the number of color channels could be 1 or 3, greatly
changing how much data the model is given.
"""

import keras #Main library
from keras import layers #Different layers for the model
from keras import optimizers #Optimization functions
from keras.utils import to_categorical #Enable use of One-Hot Labels

"""
The formula to calculate the output shape of each convolutional layer given the input shape is as follows:
    if no padding is used:
        (input shape - (kernel size - 1)) / stride length
    if padding is used:
        input shape / stride length
        
The formula to caluclate the output shape of a pooling layer given the input shape is as follows:
    input shape / kernel size

All operations are rounded down to the nearest whole number.
If no stride length is specified, the stride length is 1.
If no padding is specified, no padding is used.
The third dimension is the number of filters used, regardless of the input. (So a grayscale image and an rgb image will be the same shape in the end)
Pooling layers keep the number of filters constant.
"""

#Build the model
exampleModel = keras.Sequential(
    [
     layers.Input(shape = (210, 160, 3)), #Input shape will be a 210 x 160 rgb image (change 3 to 1 and double check the dimensions of the input image if grayscale is used)
     layers.Conv2D(32, 10, strides = 2, activation = "gelu", padding = "same"), #32 10 x 10 filters with a stride length of 2 and a padding of 0s around the edges of the image
     layers.Conv2D(32, 10, activation = "gelu"), #32 10 x 10 filters with a stride length of 1
     layers.Conv2D(64, 10, activation = "gelu"), #64 10 x 10 filters with a stride length of 1
     layers.Conv2D(64, 5, activation = "gelu"), #64 5 x 5 filters with a stride length of 1
     layers.MaxPooling2D(5), #Max Pooling using a 5 x 5 filter
     #Max pooling takes the highest value in the filter an makes it the value of a smaller "image," unlike the convolutional layers, the filter does not overlap itself
     layers.Flatten(), #Converts the data up to this point into a 1D array
     layers.Dense(256, activation = "gelu"), #Single regression layer for a small boost to feature extraction
     layers.Dense(18, activation = "softmax") #Determines the class
     #Softmax activation will bind the results to a range of [0, 1], with the sum of all nodes equaling 1
     #This allows the final layer to present a probability of each action
    ]
)

#Print a summary of each of the layers in the model
exampleModel.summary()