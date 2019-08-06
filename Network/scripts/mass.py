#!/usr/bin/env python3

from massmodel import MassModel, tryModel

import argparse
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
import random
import itertools

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
		mass = 0
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]
	return dictFiles

def to4momentum(data, Indexes):
	m_col = data[:, [Indexes['mass']]]
	px_col = np.array(data[:, [Indexes['pt']]] * np.cos(data[:, [Indexes['phi']]]))
	py_col = np.array(data[:, [Indexes['pt']]] * np.sin(data[:, [Indexes['phi']]]))
	pz_col = np.array(data[:, [Indexes['pt']]] / np.tan(2 * np.arctan(np.exp( - data[:, [Indexes['eta']]]))))
	E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)	
	return [E_col, px_col, py_col, pz_col]

def topxpy(data, Indexes):
	px_col = np.array(data[:, [Indexes['pt']]] * np.cos(data[:, [Indexes['phi']]]))
	py_col = np.array(data[:, [Indexes['pt']]] * np.sin(data[:, [Indexes['phi']]]))
	return [px_col, py_col]

def topxpypzE(data, Indexes):
	px_col = np.array(data[:, [Indexes['pt']]] * np.cos(data[:, [Indexes['phi']]]))
	py_col = np.array(data[:, [Indexes['pt']]] * np.sin(data[:, [Indexes['phi']]]))
	E_col = np.sqrt(px_col**2 + py_col**2)
	pz_col = np.zeros(px_col.shape)
	return [E_col, px_col, py_col, pz_col]

def formData(data, forConv = False):
	inputList = []

	#Transform final partilces momenta
	for numParticle in range(5):
		Indexes = {'mass': 4 * numParticle + 3,
			 		'eta': 4 * numParticle + 2,
					'phi': 4 * numParticle + 1,
					'pt' : 4 * numParticle }
		inputList += to4momentum(data, Indexes)
	
	#Transform neutrino's momentum	
	Indexes = {'phi': 4 * 5 + 1,
				'pt': 4 * 5 }
	if (forConv):
		inputList += topxpypzE(data, Indexes)
	else:
		inputList += topxpy(data, Indexes)	

	#Transform H's momentum
	Indexes = {'mass': -1, 'eta': -2, 'phi': -3, 'pt': -4}
	inputList += to4momentum(data, Indexes)		

	inputTuple = tuple(inputList)
	dataFormed = np.concatenate(inputTuple, axis = 1)
	return dataFormed

def pt_order(p4):
	return (p4[1][0]**2 + p4[2][0]**2)

def trainModel(tuneFlag, randFlag, trueHFlag):

	masses = [200, 300]

	dataFileTempEle = "data/massTraining/e4j/L4B_{}_100.csv"
	dataFileTempMu = "data/massTraining/mu4j/L4B_{}_100.csv"
	data_files = [
			"data/massTraining/e4j/L4B_200_100.csv"]
	
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

	bkg_files = [] #["data/massTraining/e4j/TT+j-1L.csv"]

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

	E_low, E_high = [0, 500]
	p_low, p_high = [-500, 500]
	forConv = True


	for mass in dictFiles:
		start_index_test = len(dataTest)
		start_index_train = len(dataTrain)
		for file_name in dictFiles[mass]:
			dataFrame = pd.read_csv(file_name, sep=",")
			dataFrame = dataFrame.sample(frac=1).reset_index(drop=True) #Shuffle dataset
			if (mass != 0):
				data = dataFrame.to_numpy()
			else:
				data = dataFrame.to_numpy()[:40000:, :]
			trainFrac = int(0.9 * data.shape[0])
			s_train = slice(0, trainFrac, 1)	
			s_test = slice(trainFrac, data.shape[0], 1)
			data = formData(data, forConv)
			if (all_input_files.index(file_name) == 0):
				dataTest, resultTest = data[s_test, :24], data[s_test, -4:]
				dataTrain, resultTrain = data[s_train, :24], data[s_train, -4:]

				if (mass != 0):
				    if (trueHFlag):
				    	resultTest = [int(file_name[-11:-8])] * data[s_test, [-1]].shape[0]
				    	resultTrain = [int(file_name[-11:-8])] * data[s_train, [-1]].shape[0]
				else:
					resultTest = data[s_test, [-1, -2, -3, -4]]
					resultTrain = data[s_train, [-1, -2, -3, -4]]

					a = np.zeros(len(resultTest))
					resultTest = np.column_stack((1000 + a, a, a, a))

					b = np.zeros(len(resultTrain))
					resultTrain = np.column_stack((1000 + b, b, b, b))
					
					"""
					resultTest[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTest))	
					resultTest[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest))
					resultTest[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest))
					resultTest[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest))

					resultTrain[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTrain))	
					resultTrain[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain))
					resultTrain[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain))
					resultTrain[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain))
					"""
			else:
				dataTest = np.append(dataTest, data[s_test, :24], axis = 0)
				dataTrain = np.append(dataTrain, data[s_train, :24], axis = 0)
				if (mass != 0):
					if (trueHFlag):
					    resultTest = np.append(resultTest, [int(file_name[-11:-8])]*data[trainFrac::, [-1]].shape[0])
					    resultTrain = np.append(resultTrain, [int(file_name[-11:-8])]*data[:trainFrac:, [-1]].shape[0])	
					else:
						resultTest = np.append(resultTest, data[s_test, -4:], axis = 0)
						resultTrain = np.append(resultTrain, data[s_train, -4:], axis = 0)
				else:
					resultTest_add = data[s_test, [-1,  -2, -3, -4]]
					resultTrain_add = data[s_train, [-1,  -2, -3, -4]]

					a = np.zeros(len(resultTest_add))
					resultTest_add = np.column_stack((1000 + a, a, a, a))

					b = np.zeros(len(resultTrain_add))
					resultTrain_add = np.column_stack((1000 + b, b, b, b))

					"""
					resultTest_add[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTest_add))	
					resultTest_add[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest_add))
					resultTest_add[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest_add))
					resultTest_add[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTest_add))

					resultTrain_add[:, 0] = np.random.uniform(low = E_low, high = E_high, size = len(resultTrain_add))	
					resultTrain_add[:, 1] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain_add))
					resultTrain_add[:, 2] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain_add))
					resultTrain_add[:, 3] = np.random.uniform(low = p_low, high = p_high, size = len(resultTrain_add))
					"""
					resultTest = np.append(resultTest, resultTest_add, axis = 0)
					resultTrain = np.append(resultTrain, resultTrain_add, axis = 0)
                    
		end_index_test = len(dataTest)
		dictMassIndTest[mass] = (start_index_test, end_index_test)
		end_index_train = len(dataTrain)
		dictMassIndTrain[mass] = (start_index_train, end_index_train)

	print(dictMassIndTrain)

	bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")
	bkgTest = bkgFrame.to_numpy()[:40000:, :-12]
	bkgTest = formData(bkgTest, forConv)[:, :-4]
	
    ##Train model/models

	##Convolutional network

	dataTrain = dataTrain.reshape(-1, 6, 4, 1)
	dataTest = dataTest.reshape(-1, 6, 4, 1)
	bkgTest = bkgTest.reshape(-1, 6, 4, 1)

	print(dataTrain[0])
	print("dataTrain.shape ", dataTrain.shape)
	dataTrain = np.array([sorted(list(event), key=pt_order, reverse=True) for event in dataTrain])
	print("dataTrain after sorting: \n", dataTrain[0])	
	dataTest = np.array([sorted(list(event), key=pt_order, reverse=True) for event in dataTest])
	bkgTest = np.array([sorted(list(event), key=pt_order, reverse=True) for event in bkgTest])

	if (tuneFlag):
		##Create a list of hyperParams
		min_layers, max_layers, step_layers = 2, 5, 1
		min_nodes, max_nodes, step_nodes = 50, 200, 50
		min_batch, max_batch, step_batch = 25, 100, 25
		activation_functions = ["relu", "elu", "selu", "softplus"]
		num_of_kernels = [4, 8, 16, 32]
		size_of_kernels = [(partDim + 1, momentaDim + 1) for partDim in range(6) for momentaDim in range(4)]
		hyperParamsSet = []
		parValues = {'nLayer': [5], #range(min_layers, max_layers + 1), 
					'activation': ["softplus"], #activation_functions, 
					'nNodes': [200], #range(min_nodes, max_nodes + 1, step_nodes), 
					'dropout': [0.3],
					'batchSize': [100], #range(min_batch, max_batch + 1, step_batch),
					'kernel': [(partDim + 1, momentaDim + 1) for partDim in range(6) for momentaDim in range(4)],
					'nKernels': [4, 8, 16, 32],
					'nEpoch': [25]}
		hyperParamsSet = list(dict(zip(parValues.keys(), values)) for values in itertools.product(*parValues.values())) 
		#result_objects = []
		cpu_count = mp.cpu_count()
		fileParamsName = "name"

		hyperParamsSet_to_test = hyperParamsSet
		results = []
		if (randFlag):
			hyperParamsSetRandom = []
			##Jobs are becoming processed here!
			for trial in range(30):
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
			result_objects = []
			if (pool_i == poolNum):
				jobsNum = parNum - (poolNum - 1) * cpu_count
			else:
				jobsNum = cpu_count
			pool = mp.Pool(jobsNum)
			for hyperParams in hyperParamsSet_to_test[start:start + jobsNum]:
				args = (dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTest)
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
					'batchSize': 101,
					'kernel': (2, 2),
					'nKernels': 16,
					'nEpoch': 100}
		modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTest)
		#modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, hyperParams)
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
