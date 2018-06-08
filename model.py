from __future__ import print_function
from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Dense, MaxPooling2D, add, Input, Concatenate, Dropout
from keras.applications.resnet50 import identity_block, conv_block


def hires_block(input_tensor, filters, block, stage):
    """
    :param input_tensor: list [convolution_path, skip_path]
    :param filters: list [skip_channel, squeeze, expand]
    :param block: number or something for name layers
    :param stage: number for name layers
    :return: convolution_path, skip_path
    """
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage))(input_tensor[0])
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage))(x)
    z = add([GlobalAveragePooling2D()(x), (input_tensor[1])])
    z = Activation('relu', name='skip_{}_relu_{}'.format(block, stage))(z)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage))(x)
    x = Conv2D(filters2, (1, 1), name='block_{}_Conv_{}'.format(block, stage+1))(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+1))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+1))(x)
    x = Conv2D(filters3, (3, 3), padding='same', name='block_{}_Conv_{}'.format(block, stage+2))(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+2))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+2))(x)
    return x, z


def hires_res_block(input_tensor, filters, block, stage):
    """
    :param input_tensor: list [convolution_path, skip_path]
    :param filters: list [skip_channel, squeeze, expand]
    :param block: number or something for name layers
    :param stage: number for name layers
    :return: convolution_path, skip_path
    """
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage))(input_tensor[0])
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage))(x)
    z = add([GlobalAveragePooling2D()(x), (input_tensor[1])])
    z = Activation('relu', name='skip_{}_relu_{}'.format(block, stage))(z)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage))(x)
    x = Conv2D(filters2, (3, 3), padding='same', name='block_{}_Conv_{}'.format(block, stage+1))(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+1))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+1))(x)
    x = Conv2D(filters3, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage+2))(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+2))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+2))(x)
    return x, z


def hires_sq_block(input_tensor, filters, block, stage):
    """
    :param input_tensor: list [convolution_path, skip_path]
    :param filters: list [skip_channel, squeeze, expand]
    :param block: number or something for name layers
    :param stage: number for name layers
    :return: convolution_path, skip_path
    """
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage))(input_tensor[0])
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage))(x)
    z = add([GlobalAveragePooling2D()(x), (input_tensor[1])])
    z = Activation('relu', name='skip_{}_relu_{}'.format(block, stage))(z)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage))(x)
    r = Conv2D(filters2, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage+1))(x)
    r = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+1))(r)
    r = Activation('relu', name='block_{}_relu_{}'.format(block, stage+1))(r)
    x = Conv2D(filters3, (3, 3), padding='same', name='block_{}_Conv_{}'.format(block, stage+2))(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+2))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+2))(x)
    return Concatenate()([r, x]), z


def hires_concat_block(input_tensor, filters, block, stage):
    """
    :param input_tensor: list [convolution_path, skip_path]
    :param filters: list [skip_channel, squeeze, expand]
    :param block: number or something for name layers
    :param stage: number for name layers
    :return: convolution_path, skip_path
    """
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage),
               kernel_initializer='he_uniform', use_bias=False)(input_tensor[0])
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage))(x)
    z = Concatenate()([GlobalAveragePooling2D()(x), (input_tensor[1])])
    z = Activation('relu', name='skip_{}_relu_{}'.format(block, stage))(z)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(filters2, (3, 3), padding='same', name='block_{}_Conv_{}'.format(block, stage+1),
               kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+1))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+1))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(filters3, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage+2),
               kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+2))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+2))(x)
    # x = Dropout(0.2)(x)
    return x, z


def hires_add_block(input_tensor, filters, block, stage):
    """
    :param input_tensor: list [convolution_path, skip_path]
    :param filters: list [skip_channel, squeeze, expand]
    :param block: number or something for name layers
    :param stage: number for name layers
    :return: convolution_path, skip_path
    """
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage),
               kernel_initializer='he_uniform', use_bias=False)(input_tensor[0])
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage))(x)
    z = add([GlobalAveragePooling2D()(x), (input_tensor[1])])
    z = Activation('relu', name='skip_{}_relu_{}'.format(block, stage))(z)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(filters2, (3, 3), padding='same', name='block_{}_Conv_{}'.format(block, stage+1),
               kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+1))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+1))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(filters3, (1, 1), padding='same', name='block_{}_Conv_{}'.format(block, stage+2),
               kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(name='block_{}_bn_{}'.format(block, stage+2))(x)
    x = Activation('relu', name='block_{}_relu_{}'.format(block, stage+2))(x)
    # x = Dropout(0.2)(x)
    return x, z


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
    z = Activation('relu', name='skip_relu_2')(z)
    x = Activation('relu', name='block_2_relu_3')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_4')(x)
    x = BatchNormalization(name='block_2_bn_4')(x)
    x = Activation('relu', name='block_2_relu_4')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_5')(x)
    x = BatchNormalization(name='block_2_bn_5')(x)
    x = Activation('relu', name='block_2_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_6')(x)
    x = BatchNormalization(name='block_2_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_3')(z)
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
    z = Activation('relu', name='skip_relu_4')(z)

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
    z = Activation('relu', name='skip_relu_5')(z)
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
    z = Activation('relu', name='skip_relu_6')(z)

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
    z = Activation('relu', name='skip_relu_2')(z)
    x = Activation('relu', name='block_2_relu_3')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_4')(x)
    x = BatchNormalization(name='block_2_bn_4')(x)
    x = Activation('relu', name='block_2_relu_4')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_5')(x)
    x = BatchNormalization(name='block_2_bn_5')(x)
    x = Activation('relu', name='block_2_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_6')(x)
    x = BatchNormalization(name='block_2_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_3')(z)
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
    z = Activation('relu', name='skip_relu_4')(z)

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
    z = Activation('relu', name='skip_relu_5')(z)
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
    z = Activation('relu', name='skip_relu_6')(z)

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
    z = Activation('relu', name='skip_relu_2')(z)
    x = Activation('relu', name='block_2_relu_3')(x)
    x = Conv2D(256, 3, padding='same', name='block_2_Conv_4')(x)
    x = BatchNormalization(name='block_2_bn_4')(x)
    x = Activation('relu', name='block_2_relu_4')(x)
    x = Conv2D(64, 3, padding='same', name='block_2_Conv_5')(x)
    x = BatchNormalization(name='block_2_bn_5')(x)
    x = Activation('relu', name='block_2_relu_5')(x)
    x = Conv2D(16, 1, padding='same', name='block_2_Conv_6')(x)
    x = BatchNormalization(name='block_2_bn_6')(x)
    z = add([z, GlobalAveragePooling2D()(x)])
    z = Activation('relu', name='skip_relu_3')(z)
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
    z = Activation('relu', name='skip_relu_4')(z)

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
    z = Activation('relu', name='skip_relu_5')(z)
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
    z = Activation('relu', name='skip_relu_6')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResD():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = MaxPooling2D(name='block_1_pool')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D()(x)

    x = Conv2D(32, 1, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)

    x, z = hires_block([x, z], [16, 32, 128], 2, 3)
    x, z = hires_block([x, z], [16, 64, 256], 2, 6)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_block([x, z], [16, 128, 512], 3, 1)
    x, z = hires_block([x, z], [16, 128, 512], 3, 4)

    x = Conv2D(16, 1, name='block_4_Conv_1')(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    z = add([GlobalAveragePooling2D()(x), z])
    z = Activation('relu', name='skip_relu_6')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResE():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = MaxPooling2D(name='block_1_pool')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D()(x)

    x = Conv2D(32, 1, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)

    x, z = hires_res_block([x, z], [16, 32, 128], 2, 3)
    x, z = hires_res_block([x, z], [16, 64, 256], 2, 6)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_res_block([x, z], [16, 128, 512], 3, 1)
    x, z = hires_res_block([x, z], [16, 128, 512], 3, 4)

    x = Conv2D(16, 1, name='block_4_Conv_1')(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    z = add([GlobalAveragePooling2D()(x), z])
    z = Activation('relu', name='skip_relu_6')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResF():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = MaxPooling2D(name='block_1_pool')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D()(x)

    x = Conv2D(32, 1, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)

    x, z = hires_res_block([x, z], [16, 32, 128], 2, 3)
    x, z = hires_res_block([x, z], [16, 64, 256], 2, 6)
    x, z = hires_res_block([x, z], [16, 64, 256], 2, 9)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_res_block([x, z], [16, 128, 512], 3, 1)
    x, z = hires_res_block([x, z], [16, 128, 512], 3, 4)
    x, z = hires_res_block([x, z], [16, 128, 512], 3, 7)

    x = Conv2D(16, 1, name='block_4_Conv_1')(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    z = add([GlobalAveragePooling2D()(x), z])
    z = Activation('relu', name='skip_relu_6')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResG():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x = Conv2D(64, 3, name='block_1_Conv_2')(x)
    x = BatchNormalization(name='block_1_bn_2')(x)
    x = Activation('relu', name='block_1_relu_2')(x)
    x = MaxPooling2D(name='block_1_pool')(x)
    x = Conv2D(16, 1, name='block_1_Conv_3')(x)
    x = BatchNormalization(name='block_1_bn_3')(x)
    x = Activation('relu', name='block_1_relu_3')(x)
    z = GlobalAveragePooling2D()(x)

    x = Conv2D(32, 1, padding='same', name='block_2_Conv_1')(x)
    x = BatchNormalization(name='block_2_bn_1')(x)
    x = Activation('relu', name='block_2_relu_1')(x)
    x = Conv2D(128, 3, padding='same', name='block_2_Conv_2')(x)
    x = BatchNormalization(name='block_2_bn_2')(x)
    x = Activation('relu', name='block_2_relu_2')(x)

    x, z = hires_sq_block([x, z], [16, 32, 128], 2, 3)
    x, z = hires_sq_block([x, z], [16, 64, 256], 2, 6)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_sq_block([x, z], [16, 128, 512], 3, 1)
    x, z = hires_sq_block([x, z], [16, 128, 512], 3, 4)

    x = Conv2D(16, 1, name='block_4_Conv_1')(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    z = add([GlobalAveragePooling2D()(x), z])
    z = Activation('relu', name='skip_relu_6')(z)

    x = Dense(10, activation='softmax', name='softmax_output')(z)
    return Model(img_input, x)


def HiResH():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 5, name='block_1_Conv_1', kernel_initializer='he_uniform', use_bias=False)(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x, z = hires_add_block([x, GlobalAveragePooling2D()(x)], [32, 32, 128], 1, 2)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5)
    x = MaxPooling2D(name='block_1_pool')(x)
    x, z = hires_concat_block([x, z], [32, 64, 256], 2, 1)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 4)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_concat_block([x, z], [64, 128, 512], 3, 1)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4)
    x = Conv2D(128, 1, name='block_4_Conv_1', kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    x = GlobalAveragePooling2D()(x)
    x = add([x, z])
    x = Dense(10, activation='softmax', name='softmax_output', use_bias=False)(x)
    return Model(img_input, x)


def HiResI():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 5, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x, z = hires_add_block([x, GlobalAveragePooling2D()(x)], [32, 32, 128], 1, 2)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3+3+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3+3+3+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3+3+3+3+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3+3+3+3+3+3)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5+3+3+3+3+3+3+3+3)
    x = MaxPooling2D(name='block_1_pool')(x)
    x, z = hires_concat_block([x, z], [32, 64, 256], 2, 1)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 4)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3+3+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3+3+3+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3+3+3+3+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3+3+3+3+3+3)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 7+3+3+3+3+3+3+3+3)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_concat_block([x, z], [64, 128, 512], 3, 1)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3+3+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3+3+3+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3+3+3+3+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3+3+3+3+3+3)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4+3+3+3+3+3+3+3+3)
    x = Conv2D(128, 1, name='block_4_Conv_1')(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    x = GlobalAveragePooling2D()(x)
    x = add([x, z])
    x = Dense(10, activation='softmax', name='softmax_output')(x)
    return Model(img_input, x)


def HiResJ():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 3, padding='same', name='block_1_Conv_1', kernel_initializer='he_uniform', use_bias=False)(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x, z = hires_add_block([x, GlobalAveragePooling2D()(x)], [32, 32, 128], 1, 2)
    x, z = hires_add_block([x, z], [32, 32, 128], 1, 5)
    x = MaxPooling2D(name='block_1_pool')(x)
    x, z = hires_concat_block([x, z], [32, 64, 256], 2, 1)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 4)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_concat_block([x, z], [64, 128, 512], 3, 1)
    x, z = hires_add_block([x, z], [128, 128, 512], 3, 4)
    x = MaxPooling2D(name='block_3_pool')(x)
    x, z = hires_concat_block([x, z], [128, 256, 1024], 4, 1)
    x, z = hires_add_block([x, z], [256, 256, 1024], 4, 4)
    x = Conv2D(256, 1, name='block_5_Conv_1', kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(name='block_5_bn_1')(x)
    x = Activation('relu', name='block_5_relu_1')(x)
    x = GlobalAveragePooling2D()(x)
    x = add([x, z])
    x = Dense(10, activation='softmax', name='softmax_output', use_bias=False)(x)
    return Model(img_input, x)


def HiResSmall():
    img_input = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, 5, name='block_1_Conv_1')(img_input)
    x = BatchNormalization(name='block_1_bn_1')(x)
    x = Activation('relu', name='block_1_relu_1')(x)
    x, z = hires_add_block([x, GlobalAveragePooling2D()(x)], [32, 32, 128], 1, 2)
    x = MaxPooling2D(name='block_1_pool')(x)
    x, z = hires_concat_block([x, z], [32, 64, 256], 2, 1)
    x, z = hires_add_block([x, z], [64, 64, 256], 2, 4)
    x = MaxPooling2D(name='block_2_pool')(x)
    x, z = hires_concat_block([x, z], [64, 128, 512], 3, 1)
    x = Conv2D(128, 1, name='block_4_Conv_1')(x)
    x = BatchNormalization(name='block_4_bn_1')(x)
    x = Activation('relu', name='block_4_relu_1')(x)
    x = GlobalAveragePooling2D()(x)
    x = add([x, z])
    x = Dense(10, activation='softmax', name='softmax_output')(x)
    return Model(img_input, x)


def ResA():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, 5, padding='valid', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResB():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(16, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = conv_block(x, 3, [32, 32, 128], stage=3, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='b')
    x = conv_block(x, 3, [64, 64, 256], stage=4, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')
