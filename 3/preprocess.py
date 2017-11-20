from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Sequential
from keras import backend as K
from keras.layers import Flatten
from scipy.misc import imresize
from glob import glob
import numpy as np
from PIL import Image
import imageio
from math import floor

slp = load_model("slp.h5")

#read images and resize them 
img_list = glob("proj3_images/Numerals/*/*.png")
labels = [path.split('/')[2] for path in img_list]

# labels = [i for i in range(10)]
# labels

img_list = [imageio.imread(img) for img in img_list]
gray = [img[:, :, 0] for img in img_list]
resized = [1-(imresize(img, (28, 28)).flatten(order='C') / 255) for img in gray ]

resized = np.array(resized)

predictions = []

for i in resized:
    predictions.append(np.argmax(slp.predict(np.reshape(i,(1,784)))))

pairs = zip(predictions,labels)

hits = [pair[0]==int(pair[1]) for pair in pairs]
# print(hits)
accuracy = len(np.nonzero(hits)[0])/len(hits)
print(accuracy)
