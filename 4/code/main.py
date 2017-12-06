import pandas as pd
from glob import glob
from os import makedirs, mkdir
from shutil import copy2
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

############## DIRECTORIES #############

train_image_dir = '../data/training'
validation_image_dir = '../data/validation'
test_image_dir = '../data/test'

############## ARGS ####################

filter_size = (3, 3)
batch_size = 16
height, width = 218, 178

if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

################ LABELS #################
try:
    model = load_model('../data/celeba.h5')
except:
    original = open('../data/list_attr_celeba.txt', 'r')
    cleaned = open('../data/cleaned.txt', 'w')
    text = original.readlines()
    converted = [line.replace('-1', '0') for line in text[1:]]
    header_old = converted[0]
    header_new = 'Filename ' + header_old
    converted[0] = header_new
    stripped = [line.replace('  ', ' ') for line in converted]
    [cleaned.write(line) for line in stripped]
    original.close()
    cleaned.close()

    makedirs('../data/test/0')
    mkdir('../data/test/1')
    makedirs('../data/training/0')
    mkdir('../data/training/1')
    makedirs('../data/validation/0')
    mkdir('../data/validation/1')

    df = pd.read_table('../data/cleaned.txt', delimiter=' ')

    labels = df['Eyeglasses']

    label_count = len(labels)

    unit = int(label_count / 10)

    img_list = sorted(glob('../data/img_align_celeba/*'))

    train_bound = 8 * unit
    validation_bound = 9 * unit

    train_labels = labels[0:train_bound]
    validation_labels = labels[train_bound + 1:validation_bound]
    test_labels = labels[validation_bound + 1:]

    train_img_names = img_list[0:train_bound]
    validation_img_names = img_list[train_bound + 1:validation_bound]
    test_img_names = img_list[validation_bound + 1:]

    for counter in range(len(train_img_names)):
        copy2(train_img_names[counter], '../data/training/' +
              str(train_labels[counter]))

    for counter in range(len(validation_img_names)):
        copy2(validation_img_names[counter], '../data/validation/' +
              str(validation_labels[counter + train_bound + 1]))

    for counter in range(len(test_img_names)):
        copy2(test_img_names[counter], '../data/test/' +
              str(test_labels[counter + validation_bound + 1]))

    ################ NETWORK #################

    model = Sequential()
    model.add(Conv2D(32, filter_size, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, filter_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, filter_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    ################ TRAINING ##############

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_image_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='binary')

    # this is a similar generator, for validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_image_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_bound // batch_size,  # change this
        epochs=50,
        validation_data=validation_generator,
        validation_steps=unit // batch_size)  # change this

    model.save('../data/celeba.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_image_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary')

score = model.evaluate_generator(
    test_generator,
    15000 / batch_size)

print(score)
