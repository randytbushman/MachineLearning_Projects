from keras import layers, models, optimizers, datasets, regularizers, callbacks


def composite_convolutional_layer(x, kernel_count, kernel_size=3, stride=1, weight_decay=0.0001, dropout=0.0):
    x = layers.Conv2D(kernel_count, kernel_size, strides=stride, padding="same",
                      kernel_initializer="he_normal", kernel_regulizer=regularizers.l2(weight_decay))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x
