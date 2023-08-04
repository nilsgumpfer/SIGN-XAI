import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.activations import linear
import requests

from signxai.methods import innvestigate


def remove_softmax(model):
    # Remove last layer's softmax
    model.layers[-1].activation = linear

    return model


def calculate_explanation_innvestigate(model, x, method='lrp.epsilon', neuron_selection=None, batchmode=False, **kwargs):
    analyzer = innvestigate.create_analyzer(method, model, **kwargs)

    if neuron_selection is None:
        neuron_selection = 'max_activation'

    if not batchmode:
        ex = analyzer.analyze(X=[x], neuron_selection=neuron_selection, **kwargs)
        expl = ex[list(ex.keys())[0]][0]

        return np.asarray(expl)
    else:
        ex = analyzer.analyze(X=x, neuron_selection=neuron_selection, **kwargs)
        expl = ex[list(ex.keys())[0]]

        return np.asarray(expl)


def load_image(img_path, expand_dims=False):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    if expand_dims:
        x = np.expand_dims(x, axis=0)

    # 'RGB'->'BGR'
    x = x[..., ::-1]

    # Zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return img, x


def download_image(path):
    r = requests.get("https://raw.githubusercontent.com/nilsgumpfer/SIGN-XAI/main/signxai/examples/7867854122_b26957e9e3_o.jpg")
    with open(path, 'wb') as f:
        f.write(r.content)

def download_model(path):
    r = requests.get("https://github.com/nilsgumpfer/SIGN-XAI/raw/main/signxai/examples/DENSEMNIST10INV.h5")
    with open(path, 'wb') as f:
        f.write(r.content)


def aggregate_and_normalize_relevancemap_rgb(R):
    # Aggregate along color channels and normalize to [-1, 1]
    a = R.sum(axis=2)
    a = normalize_heatmap(a)

    return a


def normalize_heatmap(H):
    # Normalize to [-1, 1]
    a = H / np.max(np.abs(H))

    a = np.nan_to_num(a, nan=0)

    return a
