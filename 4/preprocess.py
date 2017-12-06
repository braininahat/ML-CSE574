from glob import glob
from shutil import copy2
import pandas as pd

df = pd.read_table('data/list_attr_celeba.txt', delimiter=' ')
labels = df['Eyeglasses']

# one-hot
# labels = to_categorical(labels, num_classes=2)

label_count = len(labels)

unit = int(label_count / 10)
train_count = unit * 8
validation_count = unit * 9

train_labels = labels[0:train_count]
validation_labels = labels[train_count + 1:validation_count]
test_labels = labels[validation_count + 1:]

img_list = sorted(glob('data/img_align_celeba/*'))

img_count = len(img_list)

unit = int(img_count / 10)

train_bound = 8 * unit
validation_bound = 9 * unit

train_img_names = img_list[0:train_bound]
validation_img_names = img_list[train_bound + 1:validation_bound]
test_img_names = img_list[validation_bound + 1:]

for counter in range(len(train_img_names)):
    copy2(train_img_names[counter], 'data/training/' +
          str(train_labels[counter]))

for counter in range(len(validation_img_names)):
    copy2(validation_img_names[counter], 'data/validation/' +
          str(validation_labels[counter + train_count + 1]))

for counter in range(len(test_img_names)):
    copy2(test_img_names[counter], 'data/test/' +
          str(test_labels[counter + validation_count + 1]))
