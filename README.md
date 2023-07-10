# SIGNed explanations: Unveiling relevant features by reducing bias

This repository and python package has been published alongside the following journal article:
https://doi.org/10.1016/j.inffus.2023.101883

If you use the code from this repository in your work, please cite:
```bibtex
 @article{Gumpfer2023SIGN,
    title = {SIGNed explanations: Unveiling relevant features by reducing bias},
    author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
    journal = {Information Fusion},
    pages = {101883},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2023.101883},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253523001999}
}
```

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1566253523001999-ga1_lrg.jpg" title="Graphical Abstract" width="900px"/>

## Setup

To install the package in your environment, run:

```shell
 pip3 install signxai
```


## Usage

The below example illustrates the usage of the ```signxai``` package in combination with a VGG16 model trained on imagenet:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import load_image, aggregate_and_normalize_relevancemap_rgb, download_image

# Load model
model = VGG16(weights='imagenet')

#  Remove last layer's softmax activation (we need the raw values!)
model.layers[-1].activation = None

# Load example image
path = 'example.png'
download_image(path)
img, x = load_image(path)

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
```

(Image credit for example used in this code: Greg Gjerdingen from Willmar, USA)

## Experiments

To reproduce the experiments from our paper, please find a detailed description on https://github.com/nilsgumpfer/SIGN.
