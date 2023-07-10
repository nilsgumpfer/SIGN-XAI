import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import load_example_image, aggregate_and_normalize_relevancemap_rgb


# Load model
model = VGG16(weights='imagenet')

#  Remove last layer's softmax activation (we need the raw values!)
model.layers[-1].activation = None

# Load example image
img, x = load_example_image()

# Calculate relevancemaps
R1 = calculate_relevancemap('lrpz_epsilon_0_1_std_x', np.array(x), model)
R2 = calculate_relevancemap('lrpsign_epsilon_0_1_std_x', np.array(x), model)

# Aggregate and normalize relevancemaps for visualization
H1 = aggregate_and_normalize_relevancemap_rgb(R1)
H2 = aggregate_and_normalize_relevancemap_rgb(R2)

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, figsize=(18, 6))
axs[0].imshow(img)
axs[1].matshow(H1, cmap='seismic', clim=(-1, 1))
axs[2].matshow(H2, cmap='seismic', clim=(-1, 1))

plt.show()
