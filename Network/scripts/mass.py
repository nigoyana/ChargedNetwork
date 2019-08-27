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
import os
from sklearn import preprocessing

numParticles = 8
maxNumJets = 6
numJetsConsidered = 4
ptphietamInput = True
ptphietamOutput = True
normalize = True

def parser():
    parser = argparse.ArgumentParser(description='Program for predict charged Higgs mass', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action = "store_true", help="Train model on data")
    parser.add_argument('--tune', action = "store_true", help="Tune the hyperparameters. It will be automatically parallelised (if possible)")
    parser.add_argument('--rand', action = "store_true", help="Enable random tuning. It will be automatically parallelised (if possible)")
    parser.add_argument('--trueH', action = "store_true", help="Use 'true' H mass (not generated) in training and testing")
    return parser.parse_args()

def labelData(data_files, bkg_files):
	dictFiles = dict()

	for file_name in bkg_files:
		mass = 0
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]

	for file_name in data_files:
		mass = int(file_name[-11:-8])
		if mass in dictFiles:
			dictFiles[mass].append(file_name)
		else:
			dictFiles[mass] = [file_name]
	return dictFiles

def pt_order(p4):
	return (p4[1][0]**2 + p4[2][0]**2)

def prepareData(dataTrain, dataTest, bkgTest, hyperParams):
	forConv = False
	if (hyperParams['nConvLayer'] > 0):
		forConv = True

	if (normalize):
		dataTrain = preprocessing.scale(dataTrain)	
		dataTest = preprocessing.scale(dataTest)
		bkgTest = preprocessing.scale(bkgTest)

	if (forConv):
		dataTrain = dataTrain.reshape(len(dataTrain), -1, 4, 1)
		dataTest = dataTest.reshape(len(dataTest), -1, 4, 1)
		bkgTest = bkgTest.reshape(len(bkgTest), -1, 4, 1)
		dataTrain = np.array([sorted(list(event), key=pt_order, reverse=True) for event in dataTrain])
		dataTest = np.array([sorted(list(event), key=pt_order, reverse=True) for event in dataTest])
		bkgTest = np.array([sorted(list(event), key=pt_order, reverse=True) for event in bkgTest])	

	##LSTM
	forLSTM = False
	if (hyperParams['LSTMcells'] > 0):
		forLSTM = True

	if (forLSTM):
		if (ptphietamInput):
			dataTrain = np.column_stack((dataTrain, np.zeros([len(dataTrain), 2])))
			dataTest = np.column_stack((dataTest, np.zeros([len(dataTest), 2])))
			bkgTest = np.column_stack((bkgTest, np.zeros([len(bkgTest), 2])))
		dataTrain = dataTrain.reshape(len(dataTrain), -1, 4)
		dataTest = dataTest.reshape(len(dataTest), -1, 4)
		bkgTest = bkgTest.reshape(len(bkgTest), -1, 4)

		"""
		for i in range(len(dataTrain)):
			for j in range(len(dataTrain[i])):
				if (np.linalg.norm(dataTrain[i][j]) == 0.):
					dataTrain[i] = np.delete(dataTrain[i], j)
					print("DELETED JET")			
		"""

	print("head of dataTrain after preparation")
	print(dataTrain[0:5])
	return dataTrain, dataTest, bkgTest

def to4momentumResults(results):
	Indexes = {'pt' : 0,
			   'phi': 1,
			   'eta': 2,
			   'mass':3}
	m_col = results[:, [Indexes['mass']]]
	px_col = np.array(results[:, [Indexes['pt']]] * np.cos(results[:, [Indexes['phi']]]))
	py_col = np.array(results[:, [Indexes['pt']]] * np.sin(results[:, [Indexes['phi']]]))
	pz_col = np.array(results[:, [Indexes['pt']]] / np.tan(2 * np.arctan(np.exp( - results[:, [Indexes['eta']]]))))
	E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)
	return np.column_stack((E_col, px_col, py_col, pz_col))

def prepareResults(resultTrain, resultTest):
	if ((ptphietamOutput == False) & (ptphietamInput)):
		resultTrain = np.array(to4momentumResults(resultTrain))
		resultTest = np.array(to4momentumResults(resultTest))
	#Normalization
	resultScaling = [np.mean(resultTrain, axis = 0), np.std(resultTrain, axis = 0)]
	resultTrain = preprocessing.scale(resultTrain)
	return resultTrain, resultTest, resultScaling

def trainModel(tuneFlag, randFlag, trueHFlag):

	masses = range(200, 601, 50)

	#dataFileTempEle = "data/massTraining_more_jets_4mom/e4j/L4B_{}_100.csv"
	#dataFileTempMu = "data/massTraining_more_jets_4mom/mu4j/L4B_{}_100.csv"
	if (ptphietamInput):
		dataFileTempEle = "data/massTraining_more_jets/e4j/L4B_{}_100.csv"
		dataFileTempMu = "data/massTraining_more_jets/mu4j/L4B_{}_100.csv"
		bkgFileName = "data/massTraining_more_jets/e4j/TT+j-1L.csv"
	else:
		dataFileTempEle = "data/massTraining_more_jets_4mom/e4j/L4B_{}_100.csv"
		dataFileTempMu = "data/massTraining_more_jets_4mom/mu4j/L4B_{}_100.csv"		
		bkgFileName = "data/massTraining_more_jets_4mom/e4j/TT+j-1L.csv"

	data_files = []
	data_files = [dataFileTempEle.format(mass) for mass in masses]
	data_files += [dataFileTempMu.format(mass) for mass in masses]

	bkgFrame = pd.read_csv(bkgFileName, sep=",")
#	bkgFrame = pd.read_csv("data/massTraining/e4j/TT+j-1L.csv", sep=",")
#	bkgTest = bkgFrame.to_numpy()[:40000:, :-12]

	bkg_files = [] # [bkgFileName]

	all_input_files = bkg_files + data_files

	dictFiles = labelData(data_files, bkg_files)

    ##Read csv data to numpy
	
	dictMassIndTest = dict()
	dictMassIndTrain = dict()
	data, result = [], []
	dataTest, resultTest = [], []
	dataTrain, resultTrain = [], []

	E_low, E_high = [0, 500]
	p_low, p_high = [-500, 500]
	pt_low, pt_high = [0, 1000]
	m_low, m_high = [0, 1000]

	if (ptphietamInput):
		inputParamsNum = 4 * (numParticles - 1) + 2
	else:
		inputParamsNum = 4 * numParticles # 4 * (numParticles - 1) + 2

	inputParamsNumJets = 4 * numJetsConsidered
	inputParamsNumJetsIgnored = 4 * (maxNumJets - numJetsConsidered)	

	inputIndexes = [i for i in range(0, inputParamsNumJets, 1)] + [j for j in range(4 * maxNumJets, inputParamsNum, 1)]
	outputIndexes = [-4, -3, -2, -1] # [-4, -3, -2, -1]
	print("inputs: ", inputIndexes)
	print("outputs: ", outputIndexes)

	interpolation_masses = range(250, 551, 100)

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
				#data = dataFrame.to_numpy()[:1, :]
			trainFrac = int(0.9 * data.shape[0])
			if (mass in interpolation_masses):
				trainFrac = 0
			s_train = slice(0, trainFrac, 1)	
			s_test = slice(trainFrac, data.shape[0], 1)

			if (all_input_files.index(file_name) == 0):
				dataTest, resultTest = data[s_test, inputIndexes], data[s_test, outputIndexes]
				dataTrain, resultTrain = data[s_train, inputIndexes], data[s_train, outputIndexes]

				if (mass != 0):
				    if (trueHFlag):
				    	resultTest = [int(file_name[-11:-8])] * data[s_test, [-1]].shape[0]
				    	resultTrain = [int(file_name[-11:-8])] * data[s_train, [-1]].shape[0]
				else:
					resultTest = data[s_test, outputIndexes]
					resultTrain = data[s_train, outputIndexes]

					a = np.zeros(len(resultTest))
					resultTest = np.column_stack((a, a, a, a))
					#resultTest = np.column_stack((a+1500, a+1500))

					b = np.zeros(len(resultTrain))
					resultTrain = np.column_stack((b, b, b, b))
					#resultTrain = np.column_stack((b+1000, b+1500))
					"""
					resultTest[:, 0] = np.random.uniform(low = pt_low, high = pt_high, size = len(resultTest))	
					resultTest[:, 1] = np.random.uniform(low = m_low, high = m_high, size = len(resultTest))

					resultTrain[:, 0] = np.random.uniform(low = pt_low, high = pt_high, size = len(resultTrain))
					resultTrain[:, 1] = np.random.uniform(low = m_low, high = m_high, size = len(resultTrain))
					"""
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
				dataTest = np.append(dataTest, data[s_test, inputIndexes], axis = 0)
				dataTrain = np.append(dataTrain, data[s_train, inputIndexes], axis = 0)
				if (mass != 0):
					if (trueHFlag):
					    resultTest = np.append(resultTest, [int(file_name[-11:-8])]*data[trainFrac::, [-1]].shape[0])
					    resultTrain = np.append(resultTrain, [int(file_name[-11:-8])]*data[:trainFrac:, [-1]].shape[0])	
					else:
						resultTest = np.append(resultTest, data[s_test, outputIndexes], axis = 0)
						resultTrain = np.append(resultTrain, data[s_train, outputIndexes], axis = 0)
				else:
					resultTest_add = data[s_test, outputIndexes]
					resultTrain_add = data[s_train, outputIndexes]

					a = np.zeros(len(resultTest))
					resultTest = np.column_stack((a, a, a, a))
					#resultTest = np.column_stack((a+1500, a+1500))

					b = np.zeros(len(resultTrain))
					resultTrain = np.column_stack((b, b, b, b))
					#resultTrain = np.column_stack((b+1500, b+1500))
					"""
					resultTest_add[:, 0] = np.random.uniform(low = pt_low, high = pt_high, size = len(resultTest_add))	
					resultTest_add[:, 1] = np.random.uniform(low = m_low, high = m_high, size = len(resultTest_add))

					resultTrain_add[:, 0] = np.random.uniform(low = pt_low, high = pt_high, size = len(resultTrain_add))	
					resultTrain_add[:, 1] = np.random.uniform(low = m_low, high = m_high, size = len(resultTrain_add))
					"""
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
		if (mass not in interpolation_masses):
			dictMassIndTrain[mass] = (start_index_train, end_index_train)

	print("dictMassIndTrain: ", dictMassIndTrain)
	print("dictMassIndTest: ", dictMassIndTest)

	bkgTest = bkgFrame.to_numpy()[:40000:, inputIndexes]
	# bkgTest = bkgFrame.to_numpy()[:1, inputIndexes]
	# bkgTest = formData(bkgTest, forConv)[:, inputIndexes]

    ##Train model/models

	##Convolutional network	

	if (tuneFlag):
		##Create a list of hyperParams
		min_layers, max_layers, step_layers = 1, 7, 1
		min_conv_layers, max_conv_layers, step_conv_layers = 0, 5, 1		
		min_nodes, max_nodes, step_nodes = 50, 250, 50
		min_batch, max_batch, step_batch = 25, 250, 25
		activation_functions = ["relu", "elu", "selu", "softplus"]
		num_of_kernels = [4, 8, 16, 32]
		size_of_kernels = [(partDim + 1, momentaDim + 1) for partDim in range(numParticles) for momentaDim in range(4)]
		hyperParamsSet = []
		parValues = {'nLayer': range(min_layers, max_layers + 1), 
					'activation': activation_functions, 
					'nNodes': range(min_nodes, max_nodes + 1, step_nodes), 
					'dropout': [0.01],
					'batchSize': range(min_batch, max_batch + 1, step_batch),
					'nConvLayer': range(min_conv_layers, max_conv_layers + 1, step_conv_layers),
					'kernel': [(partDim + 1, momentaDim + 1) for partDim in range(numParticles) for momentaDim in range(4)],
					'nKernels': [4, 8, 16, 32],
					'LSTMcells': 0,
					'nEpoch': [20]}
		hyperParamsSet = list(dict(zip(parValues.keys(), values)) for values in itertools.product(*parValues.values())) 
		#result_objects = []
		cpu_count = mp.cpu_count()
		fileParamsName = "name"

		hyperParamsSet_to_test = hyperParamsSet
		results = []
		if (randFlag):
			hyperParamsSetRandom = []
			##Jobs are becoming processed here!
			for trial in range(120):
				hyperParams = random.choice(hyperParamsSet)
				print(hyperParams)
				hyperParamsSetRandom.append(hyperParams)
			hyperParamsSet_to_test = hyperParamsSetRandom	
			
		
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
				dataTrainPrepared, dataTestPrepared, bkgTestPrepared = prepareData(dataTrain, dataTest, bkgTest, hyperParams)
				resultTrainPrepared, resultTestPrepared, resultScaling = prepareResults(resultTrain, resultTest)
				args = (dataTrainPrepared, resultTrainPrepared, dataTestPrepared, resultTestPrepared, resultScaling, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTestPrepared)
				result_objects.append(pool.apply_async(tryModel, args=args))
			new_results = [r.get() for r in result_objects]
			results = results + new_results
			resultsSorted = sorted(results, reverse = False)
			fileUpdatedLog = open("hyperparams_current_log_last_rand_new.txt", "w+") 
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
					'dropout': 0.01,
					'batchSize': 100,
					'nConvLayer': 0,
					'kernel': (1, 4),
					'nKernels': 8,
					'LSTMcells': 0,
					'nEpoch': 300}
		print("bkgTest before preparation", bkgTest[0])
		dataTrain, dataTest, bkgTest = prepareData(dataTrain, dataTest, bkgTest, hyperParams)
		print("bkgTest after preparation", bkgTest[0])
		resultTrain, resultTest, resultScaling = prepareResults(resultTrain, resultTest)
		#bkgTest = False
		modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, resultScaling, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTest)
		#modelScore = tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, hyperParams, False)
		print("model: ", hyperParams)
		print("modelScore = ", modelScore)

def main():

	tfconfig = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'CPU': 2})
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	session = tf.compat.v1.Session(config=tfconfig)
	tf.keras.backend.set_session(session)

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
