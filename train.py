from __future__ import print_function
from model import HiRes
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.cifar10 import load_data
import keras

if __name__ == '__main__':
    model = HiRes()
    from keras.utils import plot_model

    plot_model(model, to_file='model.png')

    print(model.summary())
    model.compile(optimizer=sgd(lr=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    reduce_lr = ReduceLROnPlateau(monitor='acc',
                                  mode='max',
                                  min_lr=1e-11,
                                  patience=5,
                                  factor=0.2)
    checkpoint = ModelCheckpoint('weights/try-1.h5',
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
                                         brightness_range=.3,
                                         shear_range=30,
                                         zoom_range=.3,
                                         channel_shift_range=.3,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         rescale=1. / 255)
    test_generator = ImageDataGenerator(rescale=1. / 255)
    train_datagen = train_generator.flow(x_train, y_train)
    test_datagen = test_generator.flow(x_test, y_test)

    # model.fit_generator(train_datagen,
    #                     epochs=50,
    #                     validation_data=test_datagen,
    #                     callbacks=[reduce_lr, checkpoint])

    model.fit(x_train, y_train,
              epochs=100,
              batch_size=32,
              validation_data=[x_test, y_test],
              callbacks=[checkpoint, reduce_lr])
