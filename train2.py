from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
from model import *
from keras.optimizers import sgd, adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.cifar10 import load_data
import keras
import json
import numpy as np
from keras import backend as K


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))


def lr_reduce(epoch, lr):
    if epoch + 1 == 150 or epoch + 1 == 225 or epoch + 1 == 310 or epoch + 1 == 311:
        return lr / 10
    else:
        return lr


if __name__ == '__main__':
    try_no = ['24']
    models = [HiResJ]
    for i in range(len(try_no)):
        count = 0
        model = models[i]()
        print(model.summary())
        model.compile(optimizer=sgd(lr=1e-2, decay=1e-6, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        reduce_lr = LearningRateScheduler(lr_reduce, verbose=0)
        checkpoint = ModelCheckpoint('weights/try-{}.h5'.format(try_no[i]),
                                     monitor='val_acc',
                                     mode='max',
                                     save_best_only=1,
                                     save_weights_only=1)
        sgdtrack = SGDLearningRateTracker()
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
                                             zoom_range=.15,
                                             channel_shift_range=0,
                                             horizontal_flip=True,
                                             vertical_flip=True)
        train_generator.fit(x_train)
        test_generator = ImageDataGenerator()
        train_datagen = train_generator.flow(x_train, y_train, batch_size=128)
        test_datagen = test_generator.flow(x_test, y_test)

        f = model.fit_generator(train_datagen,
                                epochs=300,
                                validation_data=test_datagen,
                                callbacks=[checkpoint, reduce_lr, sgdtrack])

        # f = model.fit(x_train, y_train,
        #               epochs=300,
        #               batch_size=64,
        #               validation_data=[x_test, y_test],
        #               callbacks=[checkpoint, reduce_lr])
        with open('log/try_{}.json'.format(try_no[i]), 'w') as wr:
            json.dump(f.history.__str__(), wr)
