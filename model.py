from __future__ import print_function
from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Dense, MaxPooling2D, add, Input


def HiRes():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, 3)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    z = GlobalAveragePooling2D()(x)

    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 1, padding='same')(x)
    x = BatchNormalization()(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu')(z)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 1, padding='same')(x)
    x = BatchNormalization()(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu')(z)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 1, padding='same')(x)
    x = BatchNormalization()(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu')(z)

    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 1, padding='same')(x)
    x = BatchNormalization()(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu')(z)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 1, padding='same')(x)
    x = BatchNormalization()(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu')(z)

    x = Dense(10, activation='softmax')(z)
    return Model(img_input, x)
