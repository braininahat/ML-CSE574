import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
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

df = pd.read_table('../data/list_attr_celeba.txt', delimiter=' ')
labels = df['Eyeglasses']

# one-hot
labels = to_categorical(labels, num_classes=2)

label_count = len(labels)

unit = int(label_count / 10)
train_count = unit * 8
validation_count = unit * 9

train_labels = labels[0:train_count]
validation_labels = labels[train_count + 1:validation_count]
test_labels = labels[validation_count + 1:]

# sorted till here

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
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True)
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_image_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    validation_image_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

model.save_weights('first_try.h5')
