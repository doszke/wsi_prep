import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
import extract as e
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def mnistmt():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)

    model.evaluate(x_test, y_test)


from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def change_size(image):
    img = array_to_img(image, scale=False) #returns PIL Image
    img = img.resize((50, 50)) #resize image
    img = img.convert(mode='RGB') #makes 3 channels
    arr = img_to_array(img) #convert back to array
    return arr.astype(np.float64)


if __name__ == "__main__":
    #mnistmt()
    #exit(0)



    BASE = "G:/panda/d/test/"
    IMG = "train_img/"
    MASK = "train_mask/"

    model_name = "panda_startmodel"

    print("load data")
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    #x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    #input_shape = (50, 50, 1)

    DATA_Y = to_categorical(y_train)
    num_classes = DATA_Y.shape[1]
    print(num_classes)
    DATA_X = x_train

    DATA_X = DATA_X.reshape((60000, 28, 28, 1))
    DATA_X = np.concatenate((DATA_X, DATA_X, DATA_X), axis=3)
    DATA_X = np.concatenate((DATA_X, DATA_X), axis=2)
    DATA_X = np.concatenate((DATA_X, DATA_X), axis=1)
    print(DATA_X.shape)


    # model
    efficient_net = EfficientNetB0(
        weights='imagenet',
        input_shape=(56, 56, 3),
        include_top=False,
        pooling='max'
    )

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=10, activation='sigmoid'))
    model.summary()



    model.summary()

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    x = model.predict(np.reshape(DATA_X[1,:,:,:], (1, 56, 56, 3)))
    print(x)
    print(DATA_Y[1, :])
    print(x.shape)
    print(DATA_Y[1,:].shape)
    plt.imshow(np.reshape(DATA_X[1,:,:,:], (56, 56, 3)))
    plt.show()
    #for train, test in kfold.split(X_train, y_train):
    history = model.fit(DATA_X[0:5000, :,:,:], DATA_Y[0:5000, :], epochs=5, verbose=1, shuffle="batch", batch_size=100)
    plot_hist(history)
    #score = model.evaluate(X_train[test], y_train[test], verbose=0)
        # zapis wag itd
    #print(score)
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    #score = model.evaluate(x_test, y_test, verbose=0)
    #print("Final score %s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    x = model.predict(np.reshape(DATA_X[1,:,:,:], (1, 56, 56, 3)))
    print(x)
    print(DATA_Y[1, :])
    print(x.shape)
    print(DATA_Y[1, :].shape)

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")
