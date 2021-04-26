from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
import extract as e
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    BASE = "G:/panda/d/test/"
    IMG = "train_img/"
    MASK = "train_mask/"

    model_name = "panda_startmodel"

    print("load data")
    # load data
    X, Y = mnist.load_data()#e.load_data(BASE + IMG, BASE + MASK)

    print("split data")
    DATA_X, DATA_Y = X
    data_x, data_y = Y

    print(f"ilość kafelek: {DATA_X.shape[0]}")

    DATA_Y = to_categorical(DATA_Y)
    num_classes = DATA_Y.shape[1]
    DATA_X = DATA_X.reshape((60000, 28, 28, 1))
    DATA_X = np.concatenate((DATA_X, DATA_X, DATA_X), axis=3)
    DATA_X = np.concatenate((DATA_X, DATA_X), axis=2)
    DATA_X = np.concatenate((DATA_X, DATA_X), axis=1)
    print(DATA_X.shape)

    X_train, X_test, y_train, y_test = train_test_split(DATA_X, DATA_Y, test_size=0.2) # 80% train, 20% test

    # xvalid & kfold

    kfold = KFold(n_splits=4, shuffle=True)

    # model
    model = Sequential()
    model.add(
        EfficientNetB0(
            include_top=False,
            input_shape=DATA_X.shape[1::],
            weights="imagenet",
            classifier_activation="softmax",
            classes=num_classes
        )
    )
    model.add(Flatten())
    model.add(Dense(num_classes))

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=["accuracy"])
    for train, test in kfold.split(X_train, y_train):
        history = model.fit(X_train[train], y_train[train], epochs=5, verbose=1, shuffle="batch", batch_size=125)
        score = model.evaluate(X_train[test], y_train[test], verbose=0)
        # zapis wag itd
        print(score)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Final score %s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")
