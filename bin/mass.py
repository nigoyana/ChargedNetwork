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
    parser.add_argument('--tune', action = "store_true", help="Tune the hyperparameters")
    return parser.parse_args()


def tryModel(model, dataTrain, resultTrain, dataTest, resultTest, nEpoch, bkgTest, dictMassInd):
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
	model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
	training = model.fit(dataTrain, resultTrain, epochs=nEpoch, batch_size=25, callbacks=[callback], validation_split=0.1, verbose=2)
	model.summary()
	os.makedirs("models/" + model.title, exist_ok=True)
	path_to_save = "models/" + model.title + "/" + model.title
	model.save_weights(path_to_save, save_format='tf')
	##Check mass distribution on test data
	##masses = xrange(200, 650, 50)
	bkgPrediction = model.predict(bkgTest).flatten()
	for mass in dictMassInd:
		start = dictMassInd[mass][0]
		end = dictMassInd[mass][1]
		signalPrediction = model.predict(dataTest[start:end]).flatten()
		model.plotTraining(training, signalPrediction, bkgPrediction, resultTest[start:end], mass)


def trainModel(tuneFlag):
	data = []
	result = []

	nEpoch = 500

	dataTest, resultTest = [], []

	data_files = [
			"data/massTraining/mu4j/L4B_200_100.csv",
			"data/massTraining/mu4j/L4B_250_100.csv",
			"data/massTraining/mu4j/L4B_300_100.csv",
			"data/massTraining/mu4j/L4B_350_100.csv",
			"data/massTraining/mu4j/L4B_400_100.csv",
			"data/massTraining/mu4j/L4B_450_100.csv",
			"data/massTraining/mu4j/L4B_500_100.csv",
			"data/massTraining/mu4j/L4B_550_100.csv",
			"data/massTraining/mu4j/L4B_600_100.csv"]

	dictFiles = dict()

	for file_name in data_files:
		mass = int(file_name[-11:-8])
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]

	print("dictFiles: ", dictFiles)
    ##Read csv data to numpy
	dictMassInd = dict()
	dataTest = []
	for mass in dictFiles:
		start_index = len(dataTest)
		for file_name in dictFiles[mass]:
			dataFrame = pd.read_csv(file_name, sep=",")
			data = dataFrame.to_numpy()
			trainFrac = int(0.9 * data.shape[0])
			if (data_files.index(file_name) == 0):
				#dataTest = data[trainFrac::, :-12]
				#resultTest = [int(file_name[-11:-8])] * data[trainFrac::, [-1]].shape[0]
				#dataTrain = data[:trainFrac:, :-12]
				#resultTrain = [int(file_name[-11:-8])] * data[:trainFrac:, [-1]].shape[0]
				dataTest, resultTest = data[trainFrac::, :-12], data[trainFrac::, [-1]]
				dataTrain, resultTrain = data[:trainFrac:, :-12], data[:trainFrac:, [-1]]
			else:
				dataTest = np.append(dataTest, data[trainFrac::, :-12], axis = 0)
				resultTest = np.append(resultTest, data[trainFrac::, [-1]], axis = 0)
				#resultTest = np.append(resultTest, [int(file_name[-11:-8])] * data[trainFrac::, [-1]].shape[0])
				dataTrain = np.append(dataTrain, data[:trainFrac:, :-12], axis = 0)
				resultTrain = np.append(resultTrain, data[:trainFrac:, [-1]], axis = 0)
				#resultTrain = np.append(resultTrain, [int(file_name[-11:-8])] * data[:trainFrac:, [-1]].shape[0])
		end_index = len(dataTest)
		dictMassInd[mass] = (start_index, end_index)
		print(dictMassInd[mass])

	print(dictMassInd)
	
	bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")

	bkgTest = bkgFrame.to_numpy()[:40000:, :-12],

    ##Train model

	max_num_of_layers = 5
	max_num_of_nodes = 101
	step_nodes = 50
	activation_functions = ["relu", "elu", "linear", "selu", "softplus"]

	if (tuneFlag):
		for number_of_layers in range(2, max_num_of_layers):
			for activation in activation_functions:
				for number_of_nodes in range(75, max_num_of_nodes, step_nodes):
					model = MassModel(dataTrain.shape[-1], 3, number_of_layers, number_of_nodes, activation, 0.4)
					print("Training model " + model.title)
					tryModel(model, dataTrain, resultTrain, dataTest, resultTest, nEpoch, bkgTest, dictMassInd)
	else:
		model = MassModel(dataTrain.shape[-1], 3, 3, 150, "elu", 0.4)
		print("Training model " + model.title)
		tryModel(model, dataTrain, resultTrain, dataTest, resultTest, nEpoch, bkgTest, dictMassInd)


def main():
    ##Parser arguments
	args = parser()

    ##Train model
	if args.train:
		tuneFlag = False
		if args.tune:
			tuneFlag = True
		trainModel(tuneFlag)

if __name__ == "__main__":
    main()
