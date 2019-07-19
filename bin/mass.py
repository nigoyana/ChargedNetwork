#!/usr/bin/env python3

from massmodel import MassModel, tryModel

import argparse
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp

def parser():
    parser = argparse.ArgumentParser(description='Program for predict charged Higgs mass', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action = "store_true", help="Train model on data")
    parser.add_argument('--tune', action = "store_true", help="Tune the hyperparameters")
    parser.add_argument('--parallel', action = "store_true", help="Parallelize the tunning")
    return parser.parse_args()

def trainModel(tuneFlag, parallelizeFlag):
	data = []
	result = []

	dataTest, resultTest = [], []

	data_files = [
			"data/massTraining/e4j/L4B_200_100.csv",
			"data/massTraining/e4j/L4B_250_100.csv",
			"data/massTraining/e4j/L4B_300_100.csv",
			"data/massTraining/e4j/L4B_350_100.csv",
			"data/massTraining/e4j/L4B_400_100.csv",
			"data/massTraining/e4j/L4B_450_100.csv",
			"data/massTraining/e4j/L4B_500_100.csv",
			"data/massTraining/e4j/L4B_550_100.csv",
			"data/massTraining/e4j/L4B_600_100.csv",
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
	
	bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")
	bkgTest = bkgFrame.to_numpy()[:40000:, :-12]

    ##Train model/models

	if (tuneFlag):
		##Create a list of hyperParams
		min_layers, max_layers, step_layers = 2, 5, 1
		min_nodes, max_nodes, step_nodes = 50, 200, 50
		min_batch, max_batch, step_batch = 25, 100, 25
		activation_functions = ["relu", "elu", "selu", "softplus"]
		#activation_functions = ["relu", "elu"]
		hyperParamsSet = []
		for number_of_layers in range(min_layers, max_layers + 1):
			for activation in activation_functions:
				for number_of_nodes in range(min_nodes, max_nodes + 1, step_nodes):
					for batchSize in range(min_batch, max_batch + 1, step_batch):
						newDict = dict()
						newDict['nLayer'] = number_of_layers
						newDict['activation'] = activation
						newDict['nNodes'] = number_of_nodes
						newDict['dropout'] = 0.3
						newDict['batchSize'] = batchSize
						newDict['nEpoch'] = 500
						hyperParamsSet.append(newDict)

		##Check if the parallelising is required 
		if (parallelizeFlag):
			pool = mp.Pool(mp.cpu_count())
			results = []
			result_objects = []
			for hyperParams in hyperParamsSet:
				args = (dataTrain, resultTrain, dataTest, resultTest, dictMassInd, bkgTest, hyperParams)
				result_objects.append(pool.apply_async(tryModel, args=args))
			results = [r.get() for r in result_objects]
			pool.close()
			pool.join()

			fileParams = open("hyperparams_stat.txt", "w+")
			resultsSorted = sorted(results, reverse = False)
			for result in resultsSorted:
				index = results.index(result)
				fileParams.write(str((result, hyperParamsSet[index])) + "\n")
			fileParams.close()
				
		else:
			results = []
			for hyperParams in hyperParamsSet:
				modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassInd, bkgTest, hyperParams)
				results.append(modelScore)
			
			fileParams = open("hyperparams_stat.txt", "w+")
			resultsSorted = sorted(results, reverse = False)
			for result in resultsSorted:
				index = results.index(result)
				fileParams.write(str((result, hyperParamsSet[index])) + "\n")
			fileParams.close()

	else:
		hyperParams = {'nLayer': 3,
					'activation': "elu",
					'nNodes': 222,
					'dropout': 0.3, 
					'batchSize': 25, 
					'nEpoch': 1}
		modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassInd, bkgTest, hyperParams)
		print("model: ", hyperParams)
		print("modelScore = ", modelScore)


def main():
    ##Parser arguments
	args = parser()

    ##Train model
	if args.train:
		tuneFlag = False
		parallelizeFlag = False
		if args.tune:
			tuneFlag = True
			if args.parallel:
				parallelizeFlag = True
		Flags = {'tuneFlag': tuneFlag,
				'parallelizeFlag': parallelizeFlag
		}
		trainModel(**Flags)

if __name__ == "__main__":
    main()
