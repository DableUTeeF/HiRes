from __future__ import print_function
from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Dense, MaxPooling2D, add, Input


def HiResA():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D(name='skip_pool_1')(x)

    x = MaxPooling2D(name='maxpool_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_3')(x)
    x = BatchNormalization(name='block_2_bn_3')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='block_2_relu_3')(z)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_4')(x)
    x = BatchNormalization(name='block_2_bn_4')(x)
    x = Activation('relu', name='block_2_relu_4')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_5')(x)
    x = BatchNormalization(name='block_2_bn_5')(x)
    x = Activation('relu', name='block_2_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_6')(x)
    x = BatchNormalization(name='block_2_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_2')(z)
    x = Activation('relu', name='block_2_relu_6')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_7')(x)
    x = BatchNormalization(name='block_2_bn_7')(x)
    x = Activation('relu', name='block_2_relu_7')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_8')(x)
    x = BatchNormalization(name='block_2_bn_8')(x)
    x = Activation('relu', name='block_2_relu_8')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_9')(x)
    x = BatchNormalization(name='block_2_bn_9')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_2')(z)

    x = Activation('relu', name='block_2_relu_9')(x)
    x = MaxPooling2D(name='maxpool_2')(x)
    x = Conv2D(256, 3, padding='same', name='block_3_Conv_1')(x)
    x = BatchNormalization(name='block_3_bn_1')(x)
    x = Activation('relu', name='block_3_relu_1')(x)
    x = Conv2D(256, 3, padding='same', name='block_3_Conv_2')(x)
    x = BatchNormalization(name='block_3_bn_2')(x)
    x = Activation('relu', name='block_3_relu_2')(x)
    x = Conv2D(16, 1, padding='same', name='block_3_Conv_3')(x)
    x = BatchNormalization(name='block_3_bn_3')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_3')(z)
    x = Activation('relu', name='block_3_relu_3')(x)
    x = Conv2D(256, 3, padding='same', name='block_3_Conv_4')(x)
    x = BatchNormalization(name='block_3_bn_4')(x)
    x = Activation('relu', name='block_3_relu_4')(x)
    x = Conv2D(256, 3, padding='same', name='block_3_Conv_5')(x)
    x = BatchNormalization(name='block_3_bn_5')(x)
    x = Activation('relu', name='block_3_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_3_Conv_6')(x)
    x = BatchNormalization(name='block_3_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_4')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResB():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D(name='skip_pool_1')(x)

    x = MaxPooling2D(name='maxpool_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_3')(x)
    x = BatchNormalization(name='block_2_bn_3')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='block_2_relu_3')(z)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_4')(x)
    x = BatchNormalization(name='block_2_bn_4')(x)
    x = Activation('relu', name='block_2_relu_4')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_5')(x)
    x = BatchNormalization(name='block_2_bn_5')(x)
    x = Activation('relu', name='block_2_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_6')(x)
    x = BatchNormalization(name='block_2_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_2')(z)
    x = Activation('relu', name='block_2_relu_6')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_7')(x)
    x = BatchNormalization(name='block_2_bn_7')(x)
    x = Activation('relu', name='block_2_relu_7')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_8')(x)
    x = BatchNormalization(name='block_2_bn_8')(x)
    x = Activation('relu', name='block_2_relu_8')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_9')(x)
    x = BatchNormalization(name='block_2_bn_9')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_2')(z)

    x = Activation('relu', name='block_2_relu_9')(x)
    x = MaxPooling2D(name='maxpool_2')(x)
    x = Conv2D(512, 3, padding='same', name='block_3_Conv_1')(x)
    x = BatchNormalization(name='block_3_bn_1')(x)
    x = Activation('relu', name='block_3_relu_1')(x)
    x = Conv2D(512, 3, padding='same', name='block_3_Conv_2')(x)
    x = BatchNormalization(name='block_3_bn_2')(x)
    x = Activation('relu', name='block_3_relu_2')(x)
    x = Conv2D(16, 1, padding='same', name='block_3_Conv_3')(x)
    x = BatchNormalization(name='block_3_bn_3')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_3')(z)
    x = Activation('relu', name='block_3_relu_3')(x)
    x = Conv2D(1024, 3, padding='same', name='block_3_Conv_4')(x)
    x = BatchNormalization(name='block_3_bn_4')(x)
    x = Activation('relu', name='block_3_relu_4')(x)
    x = Conv2D(1024, 3, padding='same', name='block_3_Conv_5')(x)
    x = BatchNormalization(name='block_3_bn_5')(x)
    x = Activation('relu', name='block_3_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_3_Conv_6')(x)
    x = BatchNormalization(name='block_3_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_4')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResC():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D(name='skip_pool_1')(x)

    x = MaxPooling2D(name='maxpool_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(32, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_3')(x)
    x = BatchNormalization(name='block_2_bn_3')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='block_2_relu_3')(z)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_4')(x)
    x = BatchNormalization(name='block_2_bn_4')(x)
    x = Activation('relu', name='block_2_relu_4')(x)
    x = Conv2D(64, 3, padding='same', name='block_2_Conv_5')(x)
    x = BatchNormalization(name='block_2_bn_5')(x)
    x = Activation('relu', name='block_2_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_6')(x)
    x = BatchNormalization(name='block_2_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_2')(z)
    x = Activation('relu', name='block_2_relu_6')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_7')(x)
    x = BatchNormalization(name='block_2_bn_7')(x)
    x = Activation('relu', name='block_2_relu_7')(x)
    x = Conv2D(64, 3, padding='same', name='block_2_Conv_8')(x)
    x = BatchNormalization(name='block_2_bn_8')(x)
    x = Activation('relu', name='block_2_relu_8')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_9')(x)
    x = BatchNormalization(name='block_2_bn_9')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_2')(z)

    x = Activation('relu', name='block_2_relu_9')(x)
    x = MaxPooling2D(name='maxpool_2')(x)
    x = Conv2D(512, 3, padding='same', name='block_3_Conv_1')(x)
    x = BatchNormalization(name='block_3_bn_1')(x)
    x = Activation('relu', name='block_3_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_3_Conv_2')(x)
    x = BatchNormalization(name='block_3_bn_2')(x)
    x = Activation('relu', name='block_3_relu_2')(x)
    x = Conv2D(16, 1, padding='same', name='block_3_Conv_3')(x)
    x = BatchNormalization(name='block_3_bn_3')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_3')(z)
    x = Activation('relu', name='block_3_relu_3')(x)
    x = Conv2D(1024, 3, padding='same', name='block_3_Conv_4')(x)
    x = BatchNormalization(name='block_3_bn_4')(x)
    x = Activation('relu', name='block_3_relu_4')(x)
    x = Conv2D(256, 3, padding='same', name='block_3_Conv_5')(x)
    x = BatchNormalization(name='block_3_bn_5')(x)
    x = Activation('relu', name='block_3_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_3_Conv_6')(x)
    x = BatchNormalization(name='block_3_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_4')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)
