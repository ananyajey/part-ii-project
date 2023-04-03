import numpy as np
import os
from PIL import Image

from keras.datasets import mnist
from keras.utils import np_utils
from .dense import Dense
from .convolutional import Convolutional
from .activation import Tanh, Sigmoid
from .reshape import Reshape
from .loss import MSE, MSE_prime, binary_cross_entropy, binary_cross_entropy_prime
#from data_processing import load_data

#from data_processing import *

X = None
Y = None

network = []

def load_data(data_folderpath, split):
    folders = os.listdir(data_folderpath)
    label_dict = {i:folders.index(i) for i in folders}

    def get_one_hot(label_dict, label):
        v = np.zeros(len(label_dict))
        v[label_dict[label]] += 1
        return v
    
    x = []
    y = []

    # TODO: load into x and y

    for genre in os.listdir(data_folderpath):
        vector = get_one_hot(label_dict, genre)
        for image in os.listdir(os.path.join(data_folderpath, genre)):
            # TODO: get img array
            image = Image.open(os.path.join(data_folderpath, genre, image))
            data = np.asarray(image)
            #label = str(np.where(np.asarray(labels)==os.path.basename(os.path.normpath(data_folderpath)))[0])
            #labels[data] = label
            
            img_array = []
            x.append(data)
            y.append(vector)

    #perm = np.random.permutation(len(x))
    #x = np.asarray(x)[perm]
    #y = np.asarray(y)[perm]


    cutoff = int(len(x) * split)
    x_train = x[:cutoff]
    x_test = x[cutoff:]

    y_train = y[:cutoff]
    y_test = y[cutoff:]

    return (x_train, y_train), (x_test, y_test)





def me():
    unsplit_data_path = "C:/Users/anany\Cambridge\Part II Project\data/test"

    (x_train, x_test), (y_train, y_test) = load_data(unsplit_data_path, 0.7)

    network = [
        Convolutional((1, 385, 385), 3, 5),
        Sigmoid(),
        Reshape((5, 383, 383), (5 * 383 * 383, 1)),
        Dense(5*383*383, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]

    epochs = 20
    learning_rate = 0.1

    for e in range(epochs):
        error = 0

        temp = zip(x_train, y_train)
        for x, y in zip(x_train, y_train):
            output = x
            for layer in network:
                output = layer.forward(output)
            
            error += binary_cross_entropy(y, output)

            grad = binary_cross_entropy_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        
        error /= len(x_train)
        print(f"{e + 1}/{epochs}, error = {error}")

    

    for x, y in zip(x_test, y_test):
        output = x
        for layer in network:
            output = layer.forward(output)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")





def MNIST():

    
    def preprocess_data(x, y, limit):
        zero_index = np.where(y == 0)[0][:limit]
        one_index = np.where(y == 1)[0][:limit]
        

        all_indices = np.random.permutation(np.hstack((zero_index, one_index)))

        x, y = x[all_indices], y[all_indices]

        x = (x.reshape(len(x), 1, 28, 28)).astype("float32") / 255

        y = np_utils.to_categorical(y)
        y = y.reshape(len(y), 2, 1)

        return x, y
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)



    network = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5*26*26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]

    epochs = 20
    learning_rate = 0.1

    for e in range(epochs):
        error = 0

        temp = zip(x_train, y_train)
        for x, y in zip(x_train, y_train):
            output = x
            for layer in network:
                output = layer.forward(output)
            
            error += binary_cross_entropy(y, output)

            grad = binary_cross_entropy_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        
        error /= len(x_train)
        print(f"{e + 1}/{epochs}, error = {error}")

    

    for x, y in zip(x_test, y_test):
        output = x
        for layer in network:
            output = layer.forward(output)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")



def XOR():

    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    network = [
        Dense(2, 3),
        Tanh(),
        Dense(3, 1),
        Tanh()
    ]

    epochs = 10000
    learning_rate = 0.0001

    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.forward(output)

            error += MSE(y, output)
            gradients = MSE_prime(y, output)

            for layer in reversed(network):
                gradients = layer.backward(gradients, learning_rate)
            
            #print(gradients)

        error /= len(X)
        print('%d/%d, error=%f' % (e + 1, epochs, error))


#MNIST()