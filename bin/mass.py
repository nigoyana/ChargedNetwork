#!/usr/bin/env python3

from massmodel import MassModel, tryModel

import argparse
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
import random

def parser():
    parser = argparse.ArgumentParser(description='Program for predict charged Higgs mass', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action = "store_true", help="Train model on data")
    parser.add_argument('--tune', action = "store_true", help="Tune the hyperparameters. It will be automatically parallelised (if possible)")
    parser.add_argument('--rand', action = "store_true", help="Enable random tuning. It will be automatically parallelised (if possible)")
    parser.add_argument('--trueH', action = "store_true", help="Use 'true' H mass (not generated) in training and testing")
    return parser.parse_args()

def labelData(data_files):
	dictFiles = dict()
	for file_name in data_files:
		mass = int(file_name[-11:-8])
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]
	return dictFiles

def trainModel(tuneFlag, randFlag, trueHFlag):

	masses = [200, 300]

	dataFileTempEle = "data/massTraining/e4j/L4B_{}_100.csv"
	dataFileTempMu = "data/massTraining/mu4j/L4B_{}_100.csv"

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

	dictFiles = labelData(data_files)
	
	"""
	dataFrameEle = [pd.read_csv(dataFileTempEle.format(mass), sep=",") for mass in masses]
	dataFrameMu = [pd.read_csv(dataFileTempMu.format(mass), sep=",") for mass in masses]

	data = pd.concenate(dataFrameEle + dataFrameMu).sample(frac=1).to_numpy()

	trainFrac = int(0.9 * data.shape[0])
	
	dataTest = data[trainFrac::, :-12]
	dataTrain = data[:trainFrac:, :-12]]

	resultTest = data[trainFrac::, [-1]]
	resultTrain = data[:trainFrac:, [-1]]
	"""
    ##Read csv data to numpy
	
	dictMassIndTest = dict()
	dictMassIndTrain = dict()
	data, result = [], []
	dataTest, resultTest = [], []
	dataTrain, resultTrain = [], []
	for mass in dictFiles:
		start_index_test = len(dataTest)
		start_index_train = len(dataTrain)
		for file_name in dictFiles[mass]:
			dataFrame = pd.read_csv(file_name, sep=",")
			data = dataFrame.to_numpy()
			trainFrac = int(0.9 * data.shape[0])
			if (data_files.index(file_name) == 0):
				dataTest = data[trainFrac::, :-12]
				dataTrain = data[:trainFrac:, :-12]
				if (trueHFlag):
					resultTest = [int(file_name[-11:-8])] * data[trainFrac::, [-1]].shape[0]
					resultTrain = [int(file_name[-11:-8])] * data[:trainFrac:, [-1]].shape[0]
				else:
					m_col = data[trainFrac::, [-1]]
					px_col = np.array(data[trainFrac::, [-4]] * np.cos(data[trainFrac::, [-3]]))
					py_col = np.array(data[trainFrac::, [-4]] * np.sin(data[trainFrac::, [-3]]))
					pz_col = np.array(data[trainFrac::, [-4]] / np.tan(2 * np.arctan(np.exp( - data[trainFrac::, [-2]]))))
					E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)

					resultTest = np.append(E_col, px_col, axis = 1)
					resultTest = np.append(resultTest, py_col, axis = 1)
					resultTest = np.append(resultTest, pz_col, axis = 1)

					#print(resultTest[:10])

					m_col = data[:trainFrac:, [-1]]
					px_col = np.array(data[:trainFrac:, [-4]] * np.cos(data[:trainFrac:, [-3]]))
					py_col = np.array(data[:trainFrac:, [-4]] * np.sin(data[:trainFrac:, [-3]]))
					pz_col = np.array(data[:trainFrac:, [-4]] / np.tan(2 * np.arctan(np.exp( - data[:trainFrac:, [-2]]))))
					E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)

					resultTrain = np.append(E_col, px_col, axis = 1)
					resultTrain = np.append(resultTrain, py_col, axis = 1)
					resultTrain = np.append(resultTrain, pz_col, axis = 1)
	
					#print(resultTrain[:10])

					resultTest = data[trainFrac::, [-1, -2, -3, -4]] ##-5 -9
					resultTrain = data[:trainFrac:, [-1, -2, -3, -4]]
			else:
				dataTest = np.append(dataTest, data[trainFrac::, :-12], axis = 0)
				dataTrain = np.append(dataTrain, data[:trainFrac:, :-12], axis = 0)
				if (trueHFlag):
					resultTest = np.append(resultTest, [int(file_name[-11:-8])]*data[trainFrac::, [-1]].shape[0])
					resultTrain = np.append(resultTrain, [int(file_name[-11:-8])]*data[:trainFrac:, [-1]].shape[0])
				else:
					m_col = data[trainFrac::, [-1]]
					px_col = np.array(data[trainFrac::, [-4]] * np.cos(data[trainFrac::, [-3]]))
					py_col = np.array(data[trainFrac::, [-4]] * np.sin(data[trainFrac::, [-3]]))
					pz_col = np.array(data[trainFrac::, [-4]] / np.tan(2 * np.arctan(np.exp( - data[trainFrac::, [-2]]))))
					E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)

					resultTest_add = np.append(E_col, px_col, axis = 1)
					resultTest_add = np.append(resultTest_add, py_col, axis = 1)
					resultTest_add = np.append(resultTest_add, pz_col, axis = 1)

					m_col = data[:trainFrac:, [-1]]
					px_col = np.array(data[:trainFrac:, [-4]] * np.cos(data[:trainFrac:, [-3]]))
					py_col = np.array(data[:trainFrac:, [-4]] * np.sin(data[:trainFrac:, [-3]]))
					pz_col = np.array(data[:trainFrac:, [-4]] / np.tan(2 * np.arctan(np.exp( - data[:trainFrac:, [-2]]))))
					E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)

					resultTrain_add = np.append(E_col, px_col, axis = 1)
					resultTrain_add = np.append(resultTrain_add, py_col, axis = 1)
					resultTrain_add = np.append(resultTrain_add, pz_col, axis = 1)
					
					resultTest = np.append(resultTest, resultTest_add, axis = 0)
					resultTrain = np.append(resultTrain, resultTrain_add, axis = 0)		

					#resultTest = np.append(resultTest, data[trainFrac::, [-1,  -2, -3, -4]], axis = 0)
					#resultTrain = np.append(resultTrain, data[:trainFrac:, [-1,  -2, -3, -4]], axis = 0)
					#print(len(resultTest))	
					#print(len(resultTrain))			
		end_index_test = len(dataTest)
		dictMassIndTest[mass] = (start_index_test, end_index_test)
		end_index_train = len(dataTrain)
		dictMassIndTrain[mass] = (start_index_train, end_index_train)


	bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")
	bkgTest = bkgFrame.to_numpy()[:40000:, :-12]

    ##Train model/models

	if (tuneFlag):
		##Create a list of hyperParams
		min_layers, max_layers, step_layers = 2, 5, 1
		min_nodes, max_nodes, step_nodes = 50, 200, 50
		min_batch, max_batch, step_batch = 25, 100, 25
		activation_functions = ["relu", "elu", "selu", "softplus"]
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
						newDict['nEpoch'] = 20
						hyperParamsSet.append(newDict)

		result_objects = []
		cpu_count = mp.cpu_count()
		fileParamsName = "name"

		hyperParamsSet_to_test = hyperParamsSet
		results = []
		if (randFlag):
			hyperParamsSetRandom = []
			##Jobs are becoming processed here!
			for trial in range(50):
				hyperParams = random.choice(hyperParamsSet)
				hyperParamsSetRandom.append(hyperParams)
			hyperParamsSet_to_test = hyperParamsSetRandom
			fileParamsName = "hyperparams_rand_stat.txt"
		else:
			fileParamsName = "hyperparams_full_stat.txt"	
			
		parNum = len(hyperParamsSet_to_test)
		poolNum = parNum//cpu_count + int(parNum%cpu_count > 0)
		start = 0
		for pool_i in range(1, poolNum + 1):
			if (pool_i == poolNum):
				jobsNum = parNum - (poolNum - 1) * cpu_count
			else:
				jobsNum = cpu_count
			pool = mp.Pool(jobsNum)
			for hyperParams in hyperParamsSet_to_test[start:start + jobsNum]:
				args = (dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, bkgTest, hyperParams)
				result_objects.append(pool.apply_async(tryModel, args=args))
			new_results = [r.get() for r in result_objects]
			results = results + new_results
			resultsSorted = sorted(results, reverse = False)
			fileUpdatedLog = open("hyperparams_current_log_last.txt", "w+") 
			for result in resultsSorted:
				print("WRITING TO HYPERPARAMS FILE")
				index = results.index(result)
				fileUpdatedLog.write(str((result, hyperParamsSet_to_test[index])) + "\n")
			fileUpdatedLog.close()				
			start += jobsNum
	else:
		hyperParams = {'nLayer': 5,
					'activation': "softplus",
					'nNodes': 200,
					'dropout': 0.3, 
					'batchSize': 103, 
					'nEpoch': 200}
		modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, bkgTest, hyperParams)
		print("model: ", hyperParams)
		print("modelScore = ", modelScore)


def main():

    ##Parser arguments
	args = parser()
	tf.compat.v1.enable_eager_execution()

	#print(tf.executing_eagerly())
    ##Train model
	if args.train:
		tuneFlag = False
		randFlag = False
		trueHFlag = False
		if args.tune:
			tuneFlag = True
			if args.rand:
				randFlag = True
		if args.trueH:
			trueHFlag = True
		Flags = {'tuneFlag': tuneFlag,
				'randFlag': randFlag,
				'trueHFlag': trueHFlag
					}
		trainModel(**Flags)

if __name__ == "__main__":
    main()
