import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data_10.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_Data():
    # load data
    X, y = load_data(DATA_PATH)
    X_new = X.reshape(X.shape[0], -1)
    with open('New4/GenreInput.csv', 'w') as f:
        np.savetxt(f, X_new, delimiter=",")
    with open('New4/GenreOutput.csv', 'w') as f:
        np.savetxt(f, y, delimiter=",")

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_Validation, y_train, y_Validation = train_test_split(X_train, y_train, test_size=0.2)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_Validation = X_Validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]

    return X_train, X_Validation, X_test, y_train, y_Validation, y_test


def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model



X_train, X_Validation, X_test, y_train, y_Validation, y_test = prepare_Data()
'''
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = build_model(input_shape)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_Validation, y_Validation), batch_size=32, epochs=10)
model.save('model4')

test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy on test set is: {}'.format(test_accuracy))
'''
