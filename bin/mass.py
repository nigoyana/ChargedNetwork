#!/usr/bin/env python3

from massmodel import MassModel

import argparse
import yaml
import os

import numpy as np
import pandas as pd
import tensorflow as tf

def parser():
    parser = argparse.ArgumentParser(description='Program for predict charged Higgs mass', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action = "store_true", help="Train model on data")

    return parser.parse_args()

def trainModel():
    data = []
    result = []

    nEpoch = 1000

    dataTest, resultTest = [], []
	
    data_files = ["data/massTraining/e4j/L4B_200_100.csv", 
		  "data/massTraining/e4j/L4B_400_100.csv", 
		  "data/massTraining/e4j/L4B_600_100.csv",
		  "data/massTraining/mu4j/L4B_200_100.csv", 
		  "data/massTraining/mu4j/L4B_400_100.csv", 
		  "data/massTraining/mu4j/L4B_600_100.csv"]

    ##Read csv data to numpy
    for file_name in data_files:
        dataFrame = pd.read_csv(file_name, sep=",")
        data = dataFrame.to_numpy()
        trainFrac = int(0.9 * data.shape[0])
        if (data_files.index(file_name) == 0):
            dataTest, resultTest = data[trainFrac::, :-12],  data[trainFrac::, [-1]]
            dataTrain, resultTrain = data[:trainFrac:, :-12], data[:trainFrac:, [-1]]
        else:
            dataTest = np.append(dataTest, data[trainFrac::, :-12], axis = 0)
            resultTest = np.append(resultTest, data[trainFrac::, [-1]], axis = 0)	
            dataTrain = np.append(dataTrain, data[:trainFrac:, :-12], axis = 0)
            resultTrain = np.append(resultTrain, data[:trainFrac:, [-1]], axis = 0)

    bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")

    bkgTest = bkgFrame.to_numpy()[:40000:, :-12],

    ##Train model

    max_num_of_layers = 5
    max_num_of_nodes = 101
    step_nodes = 50
    activation_functions = ["relu", "elu", "linear", "selu", "softplus"]

    for number_of_layers in range(2, max_num_of_layers):
        for activation in activation_functions:
            for number_of_nodes in range(50, max_num_of_nodes, step_nodes):
                model = MassModel(dataTrain.shape[-1], 3, number_of_layers, number_of_nodes, activation, 0.4)
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
                model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
                training = model.fit(dataTrain, resultTrain, epochs=nEpoch, batch_size=25, callbacks=[callback], validation_split=0.1, verbose=2)
                model.summary()
                path_to_save = "models/" + model.title + "/" + model.title
                os.makedirs("models/" + model.title, exist_ok=True)
                model.save_weights(path_to_save, save_format='tf')
                ##Check mass distribution on test data
                signalPrediction = model.predict(dataTest).flatten()
                bkgPrediction = model.predict(bkgTest).flatten()
                model.plotTraining(training, signalPrediction, bkgPrediction, resultTest)

def main():
    ##Parser arguments
    args = parser()

    ##Train model
    if args.train:         
          trainModel()  

if __name__ == "__main__":
    main()
