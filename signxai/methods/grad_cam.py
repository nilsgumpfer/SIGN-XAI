"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.
"""
import PIL.Image
from matplotlib.cm import get_cmap

"""
Adapted from Deep Learning with Python (2017).
## Setup
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras


def calculate_grad_cam_relevancemap(x, model, last_conv_layer_name, neuron_selection=None, resize=False, **kwargs):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        if neuron_selection is None:
            neuron_selection = tf.argmax(preds[0])
        class_channel = preds[:, neuron_selection]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the relevancemap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    relevancemap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    relevancemap = tf.squeeze(relevancemap)

    # Relu (filter positve values)
    relevancemap = tf.maximum(relevancemap, 0)

    # For visualization purpose, we will also normalize the relevancemap between 0 & 1
    relevancemap = relevancemap / tf.math.reduce_max(relevancemap)

    if resize is True:
        h = np.array(relevancemap.numpy())
        h = np.expand_dims(h, axis=2)
        h = np.concatenate([h for _ in range(x.shape[3])], axis=2)

        ha = keras.preprocessing.image.array_to_img(h)
        ha = ha.resize((x.shape[1], x.shape[2]))
        h2 = keras.preprocessing.image.img_to_array(ha)

        return h2
    else:
        return relevancemap.numpy()
