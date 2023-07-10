import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16

from signxai.methods.wrappers import calculate_relevancemap
import matplotlib.pyplot as plt

from signxai.utils.utils import load_image, aggregate_and_normalize_relevancemap_rgb


def run():
    # Load model
    model = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load image
    x = load_image('../data/7867854122_b26957e9e3_o.jpg')

    # Calculate explanation
    R = calculate_relevancemap('lrpsign_epsilon_0_25_std_x_mu_0', np.array(x), model)

    # Aggregate and normalize relevancemap for visualization
    H = aggregate_and_normalize_relevancemap_rgb(R)
    plt.matshow(H, cmap='seismic', clim=(-1, 1))
    plt.show()


if __name__ == '__main__':
    run()
