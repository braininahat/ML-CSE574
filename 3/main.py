import keras
import numpy as np
import imageio
from glob import glob
from keras.datasets import mnist as mnist_keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
from scipy.misc import imresize
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


train_samples = 60000
test_samples = 10000
batch_size = 128
num_classes = 10
epochs = 20
epochs_cnn = 12
img_rows = 28
img_cols = 28
logistic_trained = []

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist_keras.load_data()

def logistic():
    global logistic_trained
    print("Starting logistic regression for MNIST\n")

    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=test_samples)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Turn up tolerance for faster convergence
    clf = LogisticRegression(C=50. / train_samples,
                             multi_class='multinomial',
                             penalty='l1', solver='saga', tol=0.01)
    clf.fit(X_train, y_train)

    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(X_test, y_test)
    logistic_trained = clf
    # print('Best C % .4f' % clf.C_)
    # print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    # print("Test score with L1 penalty: %.4f" % score)
    return score

def slp(mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test):
    print("\nStarting single layer perceptron for MNIST\n")

    try:
        load_model("slp.h5")
    except:
        x_train = mnist_x_train.reshape(train_samples, 784)
        x_test = mnist_x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(mnist_y_train, num_classes)
        y_test = keras.utils.to_categorical(mnist_y_test, num_classes)

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        # model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu')) # TODO check after removing this
        # model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save("slp.h5")
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        return score[1]

def cnn(mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test):
    print("Starting convolutional neural network for MNIST\n")
    try:
        load_model("cnn.h5")
    except:

        # the data, shuffled and split between train and test sets

        if K.image_data_format() == 'channels_first':
            x_train = mnist_x_train.reshape(mnist_x_train.shape[0], 1, img_rows, img_cols)
            x_test = mnist_x_test.reshape(mnist_x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = mnist_x_train.reshape(mnist_x_train.shape[0], img_rows, img_cols, 1)
            x_test = mnist_x_test.reshape(mnist_x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(mnist_y_train, num_classes)
        y_test = keras.utils.to_categorical(mnist_y_test, num_classes)


        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs_cnn,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save("cnn.h5")
        return score[1]

# now gotta preprocess usps
def usps():
    slp = load_model("slp.h5")
    cnn = load_model("cnn.h5")

    img_list = glob("proj3_images/Numerals/*/*.png")
    labels = [int(path.split('/')[2]) for path in img_list]

    img_list = [imageio.imread(img) for img in img_list]
    gray = [img[:, :, 0] for img in img_list]

    resized = [
        1 - (imresize(
            img, (img_rows,
                  img_cols)).flatten(order='C') / 255) for img in gray]

    resized = np.array(resized)

    logistic_score = logistic_trained.score(resized,labels)
    print(logistic_score)

    slp_score = slp.evaluate(
        resized, to_categorical(labels, num_classes), verbose=0)
    print(slp_score[1])

    # for cnn
    images = np.array(
        [(1 - (np.array(
            imresize(foo, (img_rows, img_cols)))) / 255) for foo in gray])

    if K.image_data_format() == 'channels_first':
            images = images.reshape(images.shape[0], 1, img_rows, img_cols)
    else:
            images = images.reshape(images.shape[0], img_rows, img_cols, 1)

    labels = to_categorical(labels, num_classes)
    cnn_score = cnn.evaluate(images, labels, verbose=0)
    print(cnn_score[1])
    return slp_score[1], cnn_score[1]

def main():
    logistic()
    slp(mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test)
    cnn(mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test)
    usps()

if __name__ == '__main__':
    main()