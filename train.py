from __future__ import print_function
from model import *
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.cifar10 import load_data
import keras
import json
import numpy as np


def lr_reduce(epoch, lr):
    if epoch + 1 == 50*3 or epoch + 1 == 75*3:
        return lr / 10
    else:
        return lr


if __name__ == '__main__':
    try_no = ['10', '11', '12']
    models = [HiResH, HiResSmall, ResA]
    for i in range(len(try_no)):
        count = 0
        model = models[i]()
        print(model.summary())
        model.compile(optimizer=sgd(lr=0.01, decay=1e-5, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        reduce_lr = LearningRateScheduler(lr_reduce)
        checkpoint = ModelCheckpoint('weights/try-{}.h5'.format(try_no[i]),
                                     monitor='val_acc',
                                     mode='max',
                                     save_best_only=1,
                                     save_weights_only=1)

        (x_train, y_train), (x_test, y_test) = load_data()
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        train_generator = ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             zca_epsilon=1e-6,
                                             rotation_range=15,
                                             width_shift_range=.15,
                                             height_shift_range=.15,
                                             shear_range=15,
                                             zoom_range=.1,
                                             channel_shift_range=.1,
                                             horizontal_flip=True,
                                             vertical_flip=True)
        train_generator.fit(x_train)
        test_generator = ImageDataGenerator()
        train_datagen = train_generator.flow(x_train, y_train)
        test_datagen = test_generator.flow(x_test, y_test)

        f = model.fit_generator(train_datagen,
                                epochs=300,
                                validation_data=test_datagen,
                                callbacks=[checkpoint])

        # f = model.fit(x_train, y_train,
        #               epochs=100,
        #               batch_size=32,
        #               validation_data=[x_test, y_test],
        #               callbacks=[checkpoint, reduce_lr])
        with open('log/try_{}.json'.format(try_no[i]), 'w') as wr:
            json.dump(f.history.__str__(), wr)
