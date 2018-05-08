from __future__ import print_function
from model import *
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.cifar10 import load_data
import keras
import json

if __name__ == '__main__':
    try_no = ['9']
    models = [HiResH]
    for i in range(len(try_no)):
        model = models[i]()
        print(model.summary())
        model.compile(optimizer=sgd(lr=0.01, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        reduce_lr = ReduceLROnPlateau(monitor='acc',
                                      mode='max',
                                      min_lr=1e-11,
                                      patience=5,
                                      factor=0.2)
        checkpoint = ModelCheckpoint('weights/try-{}.h5'.format(try_no[i]),
                                     monitor='val_acc',
                                     mode='max',
                                     save_best_only=1,
                                     save_weights_only=1)
        (x_train, y_train), (x_test, y_test) = load_data()
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        train_generator = ImageDataGenerator(rotation_range=30,
                                             width_shift_range=.30,
                                             height_shift_range=.30,
                                             shear_range=30,
                                             zoom_range=.3,
                                             channel_shift_range=.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             rescale=1. / 255)
        test_generator = ImageDataGenerator(rescale=1. / 255)
        train_datagen = train_generator.flow(x_train, y_train)
        test_datagen = test_generator.flow(x_test, y_test)

        f = model.fit_generator(train_datagen,
                                epochs=50,
                                validation_data=test_datagen,
                                callbacks=[reduce_lr, checkpoint])

        # f = model.fit(x_train, y_train,
        #               epochs=100,
        #               batch_size=32,
        #               validation_data=[x_test, y_test],
        #               callbacks=[checkpoint, reduce_lr])
        with open('log/try_{}.json'.format(try_no[i]), 'w') as wr:
            json.dump(f.history.__str__(), wr)
