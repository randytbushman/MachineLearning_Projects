from keras.layers import *


def composite_convolutional_layer(num_kernels: int, kernel_size: int | tuple[int, int], activation_function: str = 'relu',
                                  dropout_rate: float = 0.0) -> list[Conv2D, BatchNormalization, Activation, Dropout]:
    """

    :param num_kernels: number of filter kernels for this composite layer
    :param kernel_size: spatial size of filter kernel (e.g. if kernel_size=3, then kernel is 3x3 by default)
    :param activation_function: name of the activation function to use after the convolution
    :param dropout_rate: fraction of the input units to drop, which helps prevent over-fitting
    :return:
    """
    return [
        Conv2D(num_kernels, kernel_size, padding="same", kernel_initializer="he_normal"),
        BatchNormalization(),
        Activation(activation_function),
        Dropout(dropout_rate)
    ]

