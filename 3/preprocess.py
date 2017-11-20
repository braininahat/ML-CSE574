from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Sequential
from keras import backend as K
from keras.layers import Flatten
from scipy.misc import imresize
from glob import glob
import numpy as np
from PIL import Image
import imageio

slp = load_model("cnn.h5")

#read images and resize them 
img_list = glob("proj3_images/Numerals/1/*.png")
img_list = [imageio.imread(img) for img in img_list]
gray = [img[:, :, 1] for img in img_list]
resized = [imresize(img,(28,28)).flatten(order='F') for img in gray]
resized = np.array(resized)

predictions = []

for i in resized:
    predictions.append(np.argmax(slp.predict(np.reshape(i,(1,784)))))

print(predictions)