import json
import numpy as np
from math import *
import operator
import tensorflow.keras as keras

#DATA_SHAPE = (9996, 130, 13)
DATA_PATH = "predict_data_10.json"
SONGNAMES_PATH = "New4/songNames.txt"

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


def Predict(model, X_predict, y_predict):
    # X_predict = X_predict[np.newaxis,...]
    predictions = model.predict(X_predict)
    # print(predictions)
    #predicted_index = np.argmax(predictions, axis=1)
    # print("Expected Output: {}, Predicted Output: {}".format(y_predict,predicted_index))

    # with open('New4/4GenrePrediction.csv', 'w') as f:
    #     np.savetxt(f, predictions, delimiter=",")

    return predictions



#load the model
model = keras.models.load_model('model4')

X_predict, y_predict = load_data(DATA_PATH)
X_predict = X_predict[..., np.newaxis]

# X = np.loadtxt('New4/GenreInput.csv',delimiter=',')
# X = X.reshape(X.shape[0], X.shape[1]//DATA_SHAPE[2], DATA_SHAPE[2])
# X_predict = X[..., np.newaxis]
# y_predict = np.loadtxt('New4/GenreOutput.csv',delimiter=',')

with open(SONGNAMES_PATH, 'r') as f:
    songNames = json.loads(f.read())

userPredictions = Predict(model, X_predict, y_predict)


songLibrary = {}

#counter = 0
# 1. match predictions to corresponding songs
# predictions = joblib.load('UserTestSongs.prediction')
#predictions = joblib.load('predictions.data')
#rockPredictions = joblib.load('extraRock.prediction')

    # userPredictions = np.loadtxt('New4/4GenrePrediction.csv',delimiter=',')

# with open('songTitles.txt') as f:
#    for line  in f:
#        songLibrary[line.strip('\n')] = predictions[counter]
#        counter += 1
#
# add incubus + billy Talent
# rockCounter = 0
# with open('incubus.txt') as f:
#    for line  in f:
#        songLibrary[line.strip('\n')] = rockPredictions[rockCounter]
#        rockCounter += 1
# add user chosen songs

    # userCounter = 0
    # with open('userChosenSongs.txt') as f:
    #     for line in f:
    #         songLibrary[line.strip('\n')] = userPredictions[userCounter]
    #         userCounter += 1

for i in range(len(songNames)):
    songLibrary[songNames[i]] = userPredictions[i]

#print(songLibrary)
# 2. choose a query song then use similarity algorithm to return top ten similar songs

querySong = "classical.000080.wav"
querySongData = songLibrary[str(querySong)]

del songLibrary[str(querySong)]

# del songLibrary['Big Sean - How It Feel (Lyrics)']
# del songLibrary['The Game - Ali Bomaye (Explicit) ft. 2 Chainz, Rick Ross']
# del songLibrary['Kendrick Lamar - Money Trees (HD Lyrics)']
# del songLibrary['Faint (Official Video) - Linkin Park']
# del songLibrary['Wale-Miami Nights (Ambition)']
# del songLibrary['Wale - Bad Girls Club Ft. J Cole Official Video']
# 3. find top 10 closest songs

topSongs = {}

for key, value in songLibrary.items():
    # calculate distance
    dist = np.linalg.norm(querySongData-songLibrary[key])
    # store in distance directory
    topSongs[key] = dist

# order top songs by distance
sortedSongs = sorted(topSongs.items(), key=operator.itemgetter(1))
# take top 10 closest songs
sortedSongs = sortedSongs[:10]

for value in sortedSongs:
    print(value)
