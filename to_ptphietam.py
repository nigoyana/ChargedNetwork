import pandas as pd 
import numpy as np
import os

numParticles = 8

def to4momentum(data, Indexes):
	m_col = data[:, [Indexes['mass']]]
	px_col = np.array(data[:, [Indexes['pt']]] * np.cos(data[:, [Indexes['phi']]]))
	py_col = np.array(data[:, [Indexes['pt']]] * np.sin(data[:, [Indexes['phi']]]))
	pz_col = np.array(data[:, [Indexes['pt']]] / np.tan(2 * np.arctan(np.exp( - data[:, [Indexes['eta']]]))))
	E_col = np.sqrt(px_col**2 + py_col**2 + pz_col**2 + m_col**2)
	return [E_col, px_col, py_col, pz_col]

def toEpxpypz(data, Indexes):
	px_col = np.array(data[:, [Indexes['pt']]] * np.cos(data[:, [Indexes['phi']]]))
	py_col = np.array(data[:, [Indexes['pt']]] * np.sin(data[:, [Indexes['phi']]]))
	E_col = np.sqrt(px_col**2 + py_col**2)
	pz_col = np.zeros(px_col.shape)
	return [E_col, px_col, py_col, pz_col]

def to4momentaData(data, bkgFlag = False):
	inputList = []

	#Transform final partilces momenta
	for numParticle in range(numParticles - 1):
		Indexes = {'mass': 4 * numParticle + 3,
			 		'eta': 4 * numParticle + 2,
					'phi': 4 * numParticle + 1,
					'pt' : 4 * numParticle }
		inputList += to4momentum(data, Indexes)
	
	#Transform neutrino's momentum	
	Indexes = {'phi': 4 * (numParticles - 1) + 1,
				'pt': 4 * (numParticles - 1) }

	inputList += toEpxpypz(data, Indexes)

	#Transform H's momentum
	if (bkgFlag == False):
		Indexes = {'mass': -1, 'eta': -2, 'phi': -3, 'pt': -4}
		inputList += to4momentum(data, Indexes)
	else:
		columnZeros = np.zeros(len(data)).reshape((-1, 1))
		inputList += [columnZeros] * 4

	#Prediction of the previous algorithm
	#inputList += data[:, [-1]]

	inputTuple = tuple(inputList)
	dataFormed = np.concatenate(inputTuple, axis = 1)
	return dataFormed


def main():

	input_dir = "data/massTraining_more_jets/"
	output_dir = "data/massTraining_more_jets_ptphimetaInput/"

	os.makedirs(output_dir + "e4j/", exist_ok=True)
	os.makedirs(output_dir + "mu4j/", exist_ok=True)

	filenameTempEle = "e4j/L4B_{}_100.csv"
	filenameTempMu = "mu4j/L4B_{}_100.csv"
	filenameTempBkg = "TT+j-1L.csv"

	inputFileTempEle = input_dir + filenameTempEle
	inputFileTempMu = input_dir + filenameTempMu
	inputFileTempBkg = input_dir + "e4j/" + filenameTempBkg

	outputFileTempEle = output_dir + filenameTempEle
	outputFileTempMu = output_dir + filenameTempMu
	outputFileTempBkg = output_dir + "e4j/" + filenameTempBkg

	columns = []
	numJets = numParticles - 2
	for numJet in range(1, numJets + 1):
		columns += ["E_j{}".format(numJet), "px_j{}".format(numJet), "py_j{}".format(numJet), "pz_j{}".format(numJet)]
	columns += ["E_lep", "px_lep", "py_lep", "pz_lep"]
	columns += ["E_miss", "px_miss", "py_miss", "pz_miss"]
	columns += ["E_H", "px_H", "py_H", "pz_H"]

	masses = range(200, 650, 50)

	for mass in masses:
		#Electron in the final state
		filename_ele = inputFileTempEle.format(mass)
		dataFrame = pd.read_csv(filename_ele, sep=",")
		data = dataFrame.to_numpy()
		dataFormed = to4momentaData(data)
		dataFormedFrame = pd.DataFrame(data = dataFormed, columns = columns)
		filename_ele_out = outputFileTempEle.format(mass)
		dataFormedFrame.to_csv(filename_ele_out, index = False)
		print("{} is ready".format(filename_ele_out))

		#Muon in the final state
		filename_mu = inputFileTempMu.format(mass)
		dataFrame = pd.read_csv(filename_mu, sep=",")
		data = dataFrame.to_numpy()
		dataFormed = to4momentaData(data)
		dataFormedFrame = pd.DataFrame(data = dataFormed, columns = columns)
		filename_mu_out = outputFileTempMu.format(mass)
		dataFormedFrame.to_csv(filename_mu_out, index = False)
		print("{} is ready".format(filename_mu_out))

	#Bkg
	filename_bkg = inputFileTempBkg
	dataFrame = pd.read_csv(filename_bkg, sep=",")
	data = dataFrame.to_numpy()[:40000, :]
	dataFormed = to4momentaData(data, bkgFlag = True)
	dataFormedFrame = pd.DataFrame(data = dataFormed, columns = columns)
	filename_bkg_out = outputFileTempBkg
	dataFormedFrame.to_csv(filename_bkg_out, index = False)
	print("{} is ready".format(filename_bkg_out))


if __name__ == "__main__":
    main()
