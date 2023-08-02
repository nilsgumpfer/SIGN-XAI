from matplotlib import pyplot as plt
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap, calculate_explanation_innvestigate
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

def run():
    # Load train and test data
    ((train_images, train_labels), (val_images, val_labels)), ds_name = mnist.load_data(), 'digits'

    # Normalize color values (here: grey-scales)
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Expand pixel dimension (1 color channel)
    train_images = np.expand_dims(train_images, axis=3)
    val_images = np.expand_dims(val_images, axis=3)

    # Do one-hot encoding / do categorical conversion
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    # Extract number of classes from data dimensions
    nclasses = np.shape(train_labels)[1]

    # Define hyperparameters in dictionary for flexible use
    config = {'conv_layers': 2,
              'conv_filters': 64,
              'conv_kernel_size': 3,
              'conv_initializer': 'he_uniform',
              'conv_padding': 'same',
              'conv_activation_function': 'relu',
              'conv_dropout_rate': 0.1,
              'maxpool_stride': 2,
              'maxpool_kernel_size': 2,
              'fc_layers': 2,
              'fc_neurons': 100,
              'fc_activation_function': 'relu',
              'fc_initializer': 'he_uniform',
              'fc_dropout_rate': 0.1,
              'learning_rate': 0.01,
              'momentum': 0.9,
              'loss': 'categorical_crossentropy',
              'epochs': 1}

    # Define model architecture
    model = Sequential()

    # First convolutional and pooling layer
    model.add(Conv2D(input_shape=(28, 28, 1), filters=config['conv_filters'], kernel_size=config['conv_kernel_size'], padding=config['conv_padding'], activation=config['conv_activation_function'], kernel_initializer=config['conv_initializer']))
    model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

    # Convolutional layers
    for i in range(config['conv_layers']):
        model.add(Conv2D(filters=config['conv_filters'], kernel_size=config['conv_kernel_size'], padding=config['conv_padding'], activation=config['conv_activation_function'], kernel_initializer=config['conv_initializer']))
        model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

        if config['conv_dropout_rate'] > 0.0:
            model.add(Dropout(config['conv_dropout_rate']))

    # Global average pooling reduces number of dimensions
    model.add(GlobalAveragePooling2D())

    # Dense layers
    for i in range(config['fc_layers']):
        model.add(Dense(units=config['fc_neurons'], activation=config['fc_activation_function'], kernel_initializer=config['fc_initializer']))

        if config['fc_dropout_rate'] > 0.0:
            model.add(Dropout(config['fc_dropout_rate']))

    # Add last dense layer with neurons = number of classes
    model.add(Dense(units=nclasses, activation='softmax', kernel_initializer=config['fc_initializer']))

    # Compile model
    model.compile(optimizer=SGD(lr=config['learning_rate'], momentum=config['momentum']), loss=config['loss'], metrics=['accuracy'])

    # Print model architecture
    model.summary()

    # Train model
    model.fit(x=train_images, y=train_labels, epochs=config['epochs'], validation_data=(val_images, val_labels))

    # Evaluate model
    val_loss, val_acc = model.evaluate(val_images, val_labels)
    print('MNIST {} model - val. accuracy: {:.2f}'.format(ds_name, val_acc))

    # Remove softmax
    model.layers[-1].activation = None

    # Calculate relevancemaps
    x = val_images[123]
    R1 = calculate_relevancemap('lrpz_epsilon_0_1_std_x', np.array(x), model)
    R2 = calculate_relevancemap('lrpsign_epsilon_0_1_std_x', np.array(x), model)
    R3 = calculate_explanation_innvestigate(model, x, method='lrp.stdxepsilon', stdfactor=0.1, input_layer_rule='Z')
    R4 = calculate_explanation_innvestigate(model, x, method='lrp.stdxepsilon', stdfactor=0.1, input_layer_rule='SIGN')

    # Aggregate and normalize relevancemaps for visualization
    H1 = normalize_heatmap(R1)
    H2 = normalize_heatmap(R2)
    H3 = normalize_heatmap(R3)
    H4 = normalize_heatmap(R4)

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=5, figsize=(18, 6))
    axs[0].imshow(x)
    axs[1].matshow(H1, cmap='seismic', clim=(-1, 1))
    axs[2].matshow(H2, cmap='seismic', clim=(-1, 1))
    axs[3].matshow(H3, cmap='seismic', clim=(-1, 1))
    axs[4].matshow(H4, cmap='seismic', clim=(-1, 1))

    plt.title('original')

    plt.show()


run()
