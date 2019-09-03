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

#All these global variables can be moved to the options/arguments
masses = range(200, 601, 50) #Masses of charged Higgs boson in generated events
interpolation_masses = [] # Masses to check quality of interpolation; range(250, 551, 100)
maxNumJets = 6 #Maximum number of jets in .csv files
numParticles = maxNumJets + 2 #Number of particles in .csv files
numJetsConsidered = 4 #Number of jets in the input of NN
ptphietamInput = True #Format of inputs
ptphietamOutput = True #Format of outputs
normalize = True #Normalization of the inputs flag
#Output subdirectory in directory "models/"
outputDir = "final/check_code/"

#Parse options
def parser():
	parser = argparse.ArgumentParser(description='Program for predict charged Higgs mass', formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--train', action = "store_true", help="Train model on data")
	parser.add_argument('--withbkg', action = "store_true", help="Include background events into training")
	parser.add_argument('--bkgtorand', action = "store_true", help="Predict for background random vector")
	parser.add_argument('--tune', action = "store_true", help="Tune the hyperparameters. It will be automatically parallelised (if possible)")
	parser.add_argument('--rand', action = "store_true", help="Enable random tuning. It will be automatically parallelised (if possible)")
	parser.add_argument('--trueH', action = "store_true", help="Use 'true' H mass (not generated) in training and testing")
	return parser.parse_args()

#Create a dictionary: {key=mass: value=[file1, file2, ...]}
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

#Prepare data for training
def prepareData(dataTrain, dataTest, bkgTest, hyperParams):

	##Normalization of the inputs
	if (normalize):
		dataTrain = preprocessing.scale(dataTrain)	
		dataTest = preprocessing.scale(dataTest)
		if (bkgTest is not False):
			bkgTest = preprocessing.scale(bkgTest)

	##Convolutional input layers

	forConv = False
	if (hyperParams['nConvLayer'] > 0):
		forConv = True

	if (forConv):
		dataTrain = dataTrain.reshape(len(dataTrain), -1, 4, 1)
		dataTest = dataTest.reshape(len(dataTest), -1, 4, 1)
		if (bkgTest is not False):
			bkgTest = bkgTest.reshape(len(bkgTest), -1, 4, 1)
		dataTrain = np.array([sorted(list(event), key=pt_order, reverse=True) for event in dataTrain])
		dataTest = np.array([sorted(list(event), key=pt_order, reverse=True) for event in dataTest])
		if (bkgTest is not False):		
			bkgTest = np.array([sorted(list(event), key=pt_order, reverse=True) for event in bkgTest])	

	##LSTM input layers
	forLSTM = False
	if (hyperParams['LSTMcells'] > 0):
		forLSTM = True

	if (forLSTM):
		if (ptphietamInput):
			dataTrain = np.column_stack((dataTrain, np.zeros([len(dataTrain), 2])))
			dataTest = np.column_stack((dataTest, np.zeros([len(dataTest), 2])))
			if (bkgTest is not False):
				bkgTest = np.column_stack((bkgTest, np.zeros([len(bkgTest), 2])))
		dataTrain = dataTrain.reshape(len(dataTrain), -1, 4)
		dataTest = dataTest.reshape(len(dataTest), -1, 4)
		if (bkgTest is not False):
			bkgTest = bkgTest.reshape(len(bkgTest), -1, 4)

	return dataTrain, dataTest, bkgTest

#Transform to 4momentum
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

#Prepare results for training
def prepareResults(resultTrain, resultTest):
	if ((ptphietamOutput == False) & (ptphietamInput)):
		resultTrain = np.array(to4momentumResults(resultTrain))
		resultTest = np.array(to4momentumResults(resultTest))
	##Normalization of the outputs
	resultScaling = [np.mean(resultTrain, axis = 0), np.std(resultTrain, axis = 0)]
	resultTrain = preprocessing.scale(resultTrain)
	return resultTrain, resultTest, resultScaling

#Function for training
def trainModel(tuneFlag, randFlag, trueHFlag, withbkgFlag, bkgtorandFlag):

	##Choose directory for reading data and specify number of inputs
	if (ptphietamInput):
		dataFileTemp = "data/massTraining_more_jets/{}4j/L4B_{}_100.csv"
		bkgFileName = "data/massTraining_more_jets/e4j/TT+j-1L.csv"
		inputParamsNum = 4 * (numParticles - 1) + 2
	else:
		dataFileTemp = "data/massTraining_more_jets/{}4j/L4B_{}_100.csv"	
		bkgFileName = "data/massTraining_more_jets_4mom/e4j/TT+j-1L.csv"
		inputParamsNum = 4 * numParticles

	##Define input and output indexes
	inputIndexes = [i for i in range(0, 4 * numJetsConsidered)] + [j for j in range(4 * maxNumJets, inputParamsNum)]
	outputIndexes = [-4, -3, -2, -1]
	print("inputs: ", inputIndexes)
	print("outputs: ", outputIndexes)

	##Define list of files with input data
	all_input_files = []
	data_files = [dataFileTemp.format("e", mass) for mass in masses] + [dataFileTemp.format("mu", mass) for mass in masses]
	bkg_files = []
	##Specify some variables for training with background (if necessary)
	if (withbkgFlag):
		bkg_files += [bkgFileName]
		bkgTest = False
		if (bkgtorandFlag):
			if (ptphietamInput):
				E_low, E_high = [0, 500]
				p_low, p_high = [-500, 500]			
			else:
				pt_low, pt_high = [0, 1000]
				phi_low, phi_high = [-3.14, 3.14]
				eta_low, eta_high = [-10, 10]
				m_low, m_high = [0, 1000]
	else:
		bkgFrame = pd.read_csv(bkgFileName, sep=",")
		bkgTest = bkgFrame.to_numpy()[:40000:, inputIndexes]
	all_input_files = bkg_files + data_files

	##Create a dictionary: {key=mass: value=[file1, file2, ...]}
	dictFiles = labelData(data_files, bkg_files)

	##Lists of data numpy arrays
	to_concatenate_dataTest, to_concatenate_resultTest = [], []
	to_concatenate_dataTrain, to_concatenate_resultTrain = [], []
	
	##Dictionary {key=mass: value=(start, end)}; start and end are indexes in dataTrain
	dictMassIndTrain = dict()
	##Dictionary {key=mass: value=(start, end)}; start and end are indexes in dataTest
	dictMassIndTest = dict()

	end_index_test = 0
	end_index_train = 0
	
	##Iterate over all masses of charged Higgs
	for mass in dictFiles:
		start_index_test = end_index_test
		start_index_train = end_index_train
		##Iterate over all files for fixed mass of charged Higgs
		for file_name in dictFiles[mass]:
			dataFrame = pd.read_csv(file_name, sep=",")
			##Shuffle dataset to have different dataTrain and dataTest for each training
			##dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
			if (mass != 0):
				data = dataFrame.to_numpy()
			else:
				data = dataFrame.to_numpy()[:40000:, :]
			trainFrac = int(0.9 * data.shape[0])
			if (mass in interpolation_masses):
				trainFrac = 0
			##Slices for train and test
			s_train, s_test = slice(0, trainFrac, 1), slice(trainFrac, data.shape[0], 1)

			##Get dataTest and dataTrain 
			dataTest, resultTest = data[s_test, inputIndexes], data[s_test, outputIndexes]
			dataTrain, resultTrain = data[s_train, inputIndexes], data[s_train, outputIndexes]

			if (mass != 0):
			    if (trueHFlag):
			    	resultTest = [int(file_name[-11:-8])] * data[s_test, [-1]].shape[0]
			    	resultTrain = [int(file_name[-11:-8])] * data[s_train, [-1]].shape[0]
			else:
				##Results for background events
				a = np.zeros(len(resultTest))
				resultTest = np.column_stack((a, a, a, a))
				b = np.zeros(len(resultTrain))
				resultTrain = np.column_stack((b, b, b, b))

				if (bkgtorandFlag):
					sizeTest = len(resultTest)
					sizeTrain = len(resultTrain)
					if (ptphietamOutput):
						resultTrain[:, 0] = np.random.uniform(pt_low, pt_high, sizeTrain)
						resultTrain[:, 1] = np.random.uniform(phi_low, phi_high, sizeTrain)
						resultTrain[:, 2] = np.random.uniform(eta_low, eta_high, sizeTrain)
						resultTrain[:, 3] = np.random.uniform(m_low, m_high, sizeTrain)
						resultTest[:, 0] = np.random.uniform(pt_low, pt_high, sizeTest)
						resultTest[:, 1] = np.random.uniform(phi_low, phi_high, sizeTest)
						resultTest[:, 2] = np.random.uniform(eta_low, eta_high, sizeTest)
						resultTest[:, 3] = np.random.uniform(m_low, m_high, sizeTest)
					else:
						resultTrain[:, 0] = np.random.uniform(E_low, E_high, sizeTrain)	
						resultTrain[:, 1] = np.random.uniform(p_low, p_high, sizeTrain)
						resultTrain[:, 2] = np.random.uniform(p_low, p_high, sizeTrain)
						resultTrain[:, 3] = np.random.uniform(p_low, p_high, sizeTrain)
						resultTest[:, 0] = np.random.uniform(E_low, E_high, sizeTest)	
						resultTest[:, 1] = np.random.uniform(p_low, p_high, sizeTest)
						resultTest[:, 2] = np.random.uniform(p_low, p_high, sizeTest)
						resultTest[:, 3] = np.random.uniform(p_low, p_high, sizeTest)
			
			end_index_test += len(dataTest)
			end_index_train += len(dataTrain)
			to_concatenate_dataTest += [dataTest]
			to_concatenate_resultTest += [resultTest]
			to_concatenate_dataTrain += [dataTrain]
			to_concatenate_resultTrain += [resultTrain]	

		dictMassIndTest[mass] = (start_index_test, end_index_test)
		if (mass not in interpolation_masses):
			dictMassIndTrain[mass] = (start_index_train, end_index_train)

	##Create an array containing all the events for train and test
	dataTrain = np.concatenate(tuple(to_concatenate_dataTrain), axis = 0)
	dataTest = np.concatenate(tuple(to_concatenate_dataTest), axis = 0)
	resultTrain = np.concatenate(tuple(to_concatenate_resultTrain), axis = 0)
	resultTest = np.concatenate(tuple(to_concatenate_resultTest), axis = 0)
	del to_concatenate_dataTrain, to_concatenate_dataTest, to_concatenate_resultTrain, to_concatenate_resultTest

	print("dictMassIndTrain: ", dictMassIndTrain)
	print("dictMassIndTest: ", dictMassIndTest)

	#Train model(s)

	if (not tuneFlag):
		hyperParams = {'nLayer': 5,
					'activation': "softplus",
					'nNodes': 200,
					'dropout': 0.01,
					'batchSize': 100,
					'nConvLayer': 0,
					'kernel': (1, 4),
					'nKernels': 8,
					'LSTMcells': 0,
					'nEpoch': 1}
		dataTrainPrepared, dataTestPrepared, bkgTestPrepared = prepareData(dataTrain, dataTest, bkgTest, hyperParams)
		resultTrainPrepared, resultTestPrepared, resultScaling = prepareResults(resultTrain, resultTest)
		modelScore = tryModel(dataTrainPrepared, resultTrainPrepared, dataTestPrepared, resultTestPrepared, resultScaling, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTestPrepared, ptphietamOutput, outputDir = outputDir)
		print("model: ", hyperParams)
		print("modelScore = ", modelScore)
	else:
		#Create a list of hyperParams
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
		cpu_count = mp.cpu_count()
		fileParamsName = "name"

		hyperParamsSet_to_test = hyperParamsSet
		results = []
		if (randFlag):
			hyperParamsSetRandom = []
			#Jobs are becoming processed here!
			for trial in range(100):
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
				args = (dataTrainPrepared, resultTrainPrepared, dataTestPrepared, resultTestPrepared, resultScaling, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTestPrepared, ptphietamOutput, outputDir)
				result_objects.append(pool.apply_async(tryModel, args=args))
			new_results = [r.get() for r in result_objects]
			results = results + new_results
			resultsSorted = sorted(results, reverse = False)
			fileUpdatedLog = open("hyperparams_current_log_last_rand_new.txt", "w+") 
			for result in resultsSorted:
				print("Writing to logfile")
				index = results.index(result)
				fileUpdatedLog.write(str((result, hyperParamsSet_to_test[index])) + "\n")
			fileUpdatedLog.close()				
			start += jobsNum

##Main function
def main():

	tfconfig = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'CPU': 2})
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	session = tf.compat.v1.Session(config=tfconfig)
	tf.keras.backend.set_session(session)

    ##Parse arguments
	args = parser()

    ##Train model
	if args.train:
		Flags = {'tuneFlag': args.tune,
				'randFlag': args.rand,
				'trueHFlag': args.trueH,
				'withbkgFlag': args.withbkg,
				'bkgtorandFlag': args.bkgtorand}
		trainModel(**Flags)

if __name__ == "__main__":
    main()
