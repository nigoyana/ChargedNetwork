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

def labelData(data_files, bkg_files):
	dictFiles = dict()
	for file_name in data_files:
		mass = int(file_name[-11:-8])
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]

	for file_name in bkg_files:
		mass = 0 #int(file_name[-15:-4])
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]
	return dictFiles

def to4momentum(data, trainFrac, Indexes, trainFlg = True):
	if (trainFlg):
		m_col = data[:trainFrac:, [Indexes['mass']]]
		px_col = np.array(data[:trainFrac:, [Indexes['pt']]] * np.cos(data[:trainFrac:, [Indexes['phi']]]))
		py_col = np.array(data[:trainFrac:, [Indexes['pt']]] * np.sin(data[:trainFrac:, [Indexes['phi']]]))
		pz_col = np.array(data[:trainFrac:, [Indexes['pt']]] / np.tan(2 * np.arctan(np.exp( - data[:trainFrac:, [Indexes['eta']]]))))
		E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)	
	else:
		m_col = data[trainFrac::, [Indexes['mass']]]
		px_col = np.array(data[trainFrac::, [Indexes['pt']]] * np.cos(data[trainFrac::, [Indexes['phi']]]))
		py_col = np.array(data[trainFrac::, [Indexes['pt']]] * np.sin(data[trainFrac::, [Indexes['phi']]]))
		pz_col = np.array(data[trainFrac::, [Indexes['pt']]] / np.tan(2 * np.arctan(np.exp( - data[trainFrac::, [Indexes['eta']]]))))
		E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)
	return [E_col, px_col, py_col, pz_col]

def topxpy(data, trainFrac, Indexes, trainFlg = True):
	if (trainFlg):
		px_col = np.array(data[:trainFrac:, [Indexes['pt']]] * np.cos(data[:trainFrac:, [Indexes['phi']]]))
		py_col = np.array(data[:trainFrac:, [Indexes['pt']]] * np.sin(data[:trainFrac:, [Indexes['phi']]]))	
	else:
		px_col = np.array(data[trainFrac::, [Indexes['pt']]] * np.cos(data[trainFrac::, [Indexes['phi']]]))
		py_col = np.array(data[trainFrac::, [Indexes['pt']]] * np.sin(data[trainFrac::, [Indexes['phi']]]))
	return [px_col, py_col]		

def formData(data, trainFrac, trainFlg = False):
	inputList = []
	for numParticle in range(5):
		Indexes = {'mass': 4 * numParticle + 3,
					'eta': 4 * numParticle + 2,
					'phi': 4 * numParticle + 1,
					'pt' : 4 * numParticle }
		inputList += to4momentum(data, trainFrac, Indexes, trainFlg)
	
	Indexes = {'phi': 4 * 5 + 1,
					'pt': 4 * 5 }
	inputList += topxpy(data, trainFrac, Indexes, trainFlg)
	inputTuple = tuple(inputList)
	dataFormed = np.concatenate(inputTuple, axis = 1)
	return dataFormed

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

#	bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")
#	bkgTest = bkgFrame.to_numpy()[:40000:, :-12]

	bkg_files = ["data/massTraining/e4j/TT+j-1L.csv"]

	all_input_files = data_files + bkg_files

	dictFiles = labelData(data_files, bkg_files)

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

	#dataTestList, resultTestList = [], []
	#dataTrainList, resultTrainList = [], []
	
	E_low, E_high = [0, 500]
	p_low, p_high = [-500, 500]

	for mass in dictFiles:
		start_index_test = len(dataTest)
		start_index_train = len(dataTrain)
		for file_name in dictFiles[mass]:
			dataFrame = pd.read_csv(file_name, sep=",")
			if (mass != 0):
				data = dataFrame.to_numpy()
			else:
				data = dataFrame.to_numpy()[:40000:, :]
			trainFrac = int(0.9 * data.shape[0])
			if (all_input_files.index(file_name) == 0):
				#dataTest = data[trainFrac::, :-12]
				#dataTrain = data[:trainFrac:, :-12]
				dataTest = formData(data, trainFrac, trainFlg = False)
				dataTrain = formData(data, trainFrac, trainFlg = True)

				if (mass != 0):
				    if (trueHFlag):
				    	resultTest = [int(file_name[-11:-8])] * data[trainFrac::, [-1]].shape[0]
				    	resultTrain = [int(file_name[-11:-8])] * data[:trainFrac:, [-1]].shape[0]
				    else:
				    	Indexes = {'mass': -1, 'eta': -2, 'phi': -3, 'pt': -4}

				    	E, px, py, pz = to4momentum(data, trainFrac, Indexes, trainFlg = False)
				    	resultTest = np.concatenate((E, px, py, pz), axis = 1) 

				    	E, px, py, pz = to4momentum(data, trainFrac, Indexes, trainFlg = True)
				    	resultTrain = np.concatenate((E, px, py, pz), axis = 1)
				else:
					resultTest = data[trainFrac::, [-1, -2, -3, -4]]
					resultTrain = data[:trainFrac:, [-1, -2, -3, -4]]

					resultTest[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTest))	
					resultTest[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest))
					resultTest[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest))
					resultTest[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest))

					resultTrain[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTrain))	
					resultTrain[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain))
					resultTrain[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain))
					resultTrain[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain))


					#resultTest = data[trainFrac::, [-1, -2, -3, -4]] ##-5 -9
					#resultTrain = data[:trainFrac:, [-1, -2, -3, -4]]
			else:
				#dataTest = np.append(dataTest, data[trainFrac::, :-12], axis = 0)
				#dataTrain = np.append(dataTrain, data[:trainFrac:, :-12], axis = 0)
				dataTest = np.append(dataTest, formData(data, trainFrac, trainFlg = False), axis = 0)
				dataTrain = np.append(dataTrain, formData(data, trainFrac, trainFlg = True), axis = 0)
				if (mass != 0):
					if (trueHFlag):
					    resultTest = np.append(resultTest, [int(file_name[-11:-8])]*data[trainFrac::, [-1]].shape[0])
					    resultTrain = np.append(resultTrain, [int(file_name[-11:-8])]*data[:trainFrac:, [-1]].shape[0])
					else:
						E, px, py, pz = to4momentum(data, trainFrac, Indexes, trainFlg = False)
						resultTest_add = np.concatenate((E, px, py, pz), axis = 1)

						E, px, py, pz = to4momentum(data, trainFrac, Indexes, trainFlg = True)
						resultTrain_add = np.concatenate((E, px, py, pz), axis = 1)
					    
						resultTest = np.append(resultTest, resultTest_add, axis = 0)
						resultTrain = np.append(resultTrain, resultTrain_add, axis = 0)
					    #resultTest = np.append(resultTest, data[trainFrac::, [-1,  -2, -3, -4]], axis = 0)
					    #resultTrain = np.append(resultTrain, data[:trainFrac:, [-1,  -2, -3, -4]], axis = 0)
				else:
					#resultTest = np.append(resultTest, data[trainFrac::, [-1,  -2, -3, -4]], axis = 0)
					#resultTrain = np.append(resultTrain, data[:trainFrac:, [-1,  -2, -3, -4]], axis = 0)

					resultTest_add = data[trainFrac::, [-1,  -2, -3, -4]]
					resultTrain_add = data[:trainFrac:, [-1,  -2, -3, -4]]

					resultTest_add[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTest_add))	
					resultTest_add[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest_add))
					resultTest_add[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest_add))
					resultTest_add[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest_add))

					resultTrain_add[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTrain_add))	
					resultTrain_add[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain_add))
					resultTrain_add[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain_add))
					resultTrain_add[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain_add))

					resultTest = np.append(resultTest, resultTest_add, axis = 0)
					resultTrain = np.append(resultTrain, resultTrain_add, axis = 0)
                    
		end_index_test = len(dataTest)
		dictMassIndTest[mass] = (start_index_test, end_index_test)
		end_index_train = len(dataTrain)
		dictMassIndTrain[mass] = (start_index_train, end_index_train)

	print(dictMassIndTrain)

	#bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")
	#bkgTest = bkgFrame.to_numpy()[:40000:, :-12]

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
					'batchSize': 100, 
					'nEpoch': 50}
		#modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, bkgTest, hyperParams)
		modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, hyperParams)
		print("model: ", hyperParams)
		print("modelScore = ", modelScore)


def main():

    ##Parser arguments
	args = parser()

	#print(tf.executing_eagerly())
    ##Train model
	if args.train:
		Flags = {'tuneFlag': args.tune,
				'randFlag': args.rand,
				'trueHFlag': args.trueH}
		trainModel(**Flags)

if __name__ == "__main__":
    main()
