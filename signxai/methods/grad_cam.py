"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.

Adapted from Deep Learning with Python (2017).
"""

import numpy as np
import tensorflow as tf
from scipy.interpolate.interpolate import interp1d
from tensorflow import keras
from tensorflow.python.keras import Model


def calculate_grad_cam_relevancemap_timeseries(x, model, last_conv_layer_name, neuron_selection=None, resize=True):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    tmp = pooled_grads[..., tf.newaxis]
    heatmap = last_conv_layer_output @ tmp
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    if resize is True:
        h = np.array(heatmap.numpy())
        f = interp1d(x=np.arange(0, len(h)), y=h)
        hi = f(np.linspace(0, len(h) - 1, num=len(x[0])))
        hi = np.expand_dims(hi, axis=1)
        hc = np.concatenate([hi]*len(x[0][0]), axis=1)

        return hc
    else:
        return heatmap.numpy()


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
