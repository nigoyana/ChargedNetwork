import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import sklearn
from sklearn import preprocessing

def getLossWeights(dictMassIndTrain):
	dictWeights = dict()
	for mass in dictMassIndTrain:
		massEvts = 0
		interval = dictMassIndTrain[mass]
		massEvts += interval[1] - interval[0]
		dictWeights[mass] = massEvts
	return dictWeights

def custom_loss_twice_squared(y_true, y_pred):
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	error = np.sum((y_true**2 - y_pred**2)**2)
	return error

def custom_loss_expo(y_true, y_pred):
	diff = tf.math.squared_difference(y_pred, y_true)
	diff_sqrt_div = tf.math.divide(diff, tf.square(y_true))
	return tf.math.exp(diff_sqrt_div)

def custom_loss_4vector(y_true, y_pred):
	massTrueSq = y_true[:, 0]**2 - y_true[:, 1]**2 - y_true[:, 2]**2 - y_true[:, 3]**2
	massPredSq = y_pred[:, 0]**2 - y_pred[:, 1]**2 - y_pred[:, 2]**2 - y_pred[:, 3]**2
	return np.sum((massTrueSq - massPredSq)**2)

def custom_loss_relativeMSE(y_true, y_pred):
	diff = tf.math.squared_difference(y_pred, y_true)
	diff_sqrt_div = tf.math.divide(diff, tf.square(y_true))
	return diff_sqrt_div

def custom_loss_relativeMAE(y_true, y_pred):
	diff = tf.math.squared_difference(y_pred, y_true)
	diff_sqrt_div = tf.math.sqrt(tf.math.divide(diff, tf.square(y_true)))
	return diff_sqrt_div

def generate_training_data(dataTrain, resultTrain, batch_size=25):
	dataTrain, resultTrain = sklearn.utils.shuffle(dataTrain, resultTrain)
	while (True):
		ind = 0
		while (ind < dataTrain.shape[0]):
			yield (dataTrain[ind: ind+batch_size], resultTrain[ind: ind+batch_size])
			ind += batch_size

def generate_training_data_for_LSTM(dataTrain, resultTrain, batch_size=25):
	batch_size = 1
	dataTrainToPass = []
	for event in dataTrain:
		eventToAdd = []
		for particle in event[:6]:
			if (np.linalg.norm(particle) > 0):
				eventToAdd += [particle]
		for particle in event[6:]:
			eventToAdd += [particle]
		eventToAdd = np.array(eventToAdd)
		eventToAdd.reshape(1, -1, 4)
		dataTrainToPass += [eventToAdd]
	dataTrainToPass = np.array(dataTrainToPass)
	while (True):
		ind = 0
		dataTrainToPass, resultTrain = sklearn.utils.shuffle(dataTrainToPass, resultTrain)
		while (ind < dataTrain.shape[0]):
			yield (dataTrainToPass[ind: ind + batch_size][0].reshape(1, -1, 4), resultTrain[ind: ind + batch_size])
			ind += batch_size

def tryModel(dataTrain, resultTrain, dataTest, resultTest, resultScaling, dictMassIndTrain, dictMassIndTest, hyperParams, bkgTest = False, ptphietamOutput = True, outputDir = ""):
	nOutput = 4
	model = MassModel(tuple(dataTrain.shape[1:]), nOutput, **hyperParams)
	model.nOutput = nOutput

	#Set titles for outputs
	if (ptphietamOutput == True):
		model.OutputDescription = [{'xlabel': r'p$_t$(H$^\pm$)' + r', GeV/c$^{2}$',
									'xlim': (0, 1000),
									'bins': 40,
									'title': r"p$_t$ reconstruction for H$^\pm$ ",
									'title_add': "", 
									'name': "ptH"},
								   {'xlabel': r'$\varphi$(H$^\pm$)',
									'xlim': (-4,4),
									'bins': 32,
									'title': r"$\varphi$ reconstruction for H$^\pm$ ",
									'title_add': "", 
									'name': "phiH"},
								   {'xlabel': r'$\eta$(H$^\pm$)',
									'xlim': (-7., 7.),
									'bins': 56,
									'title': r"$\eta$ reconstruction for H$^\pm$ ",
									'title_add': "", 
									'name': "etaH"},
								   {'xlabel': r'm(H$^\pm$)' + ', GeV',
									'xlim': (0, 1000),
									'bins': 25,
									'title': r"Mass reconstruction of H$^\pm$ ",
									'title_add': "", 
									'name': "mH"}]
	else:
		model.OutputDescription = [{'xlabel': r'E(H$^\pm$)' + ', GeV',
									'xlim': (0, 3500),
									'bins': 35,
									'title': r"Energy reconstruction of H$^\pm$ ",
									'title_add': "", 
									'name': "EH"},
									{'xlabel': r'p$_x$(H$^\pm$)' + r', GeV/c$^{2}$',
									'xlim': (-750, 750),
									'bins': 30,
									'title': r"p$_x$ reconstruction for H$^\pm$ ",
									'title_add': "", 
									'name': "pxH"},
									{'xlabel': r'p$_y$(H$^\pm$)' + r', GeV/c$^{2}$',
									'xlim': (-750, 750),
									'bins': 30,
									'title': r"p$_y$ reconstruction for H$^\pm$ ",
									'title_add': "", 
									'name': "pyH"},
									{'xlabel': r'p$_z$(H$^\pm$)' + r', GeV/c$^{2}$',
									'xlim': (-3000, 3000),
									'bins': 40,
									'title': r"p$_z$ reconstruction for H$^\pm$ ",
									'title_add': "", 
									'name': "pzH"}]

	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
	model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

	dictWeights = getLossWeights(dictMassIndTrain)
	classes = []
	for mass in dictMassIndTrain:
		classes.append(mass)
	classes = np.array(classes)
	y = []
	for mass in dictMassIndTrain:
		interval = dictMassIndTrain[mass]
		massEvts = interval[1] - interval[0]
		y += [mass] * massEvts
	class_weight_vect = sklearn.utils.class_weight.compute_class_weight('balanced', classes, y)
	sample_weight = []
	for mass in dictMassIndTrain:
		interval = dictMassIndTrain[mass]
		massEvts = interval[1] - interval[0]
		weight = class_weight_vect[list(classes).index(mass)]
		sample_weight += [weight] * massEvts
	sample_weight = np.array(sample_weight)
	try:
		dataTrain, resultTrain = sklearn.utils.shuffle(dataTrain, resultTrain)
		training = model.fit(dataTrain, resultTrain, epochs=hyperParams['nEpoch'], batch_size=hyperParams['batchSize'], callbacks=[callback], validation_split=0.1, verbose=2, sample_weight = sample_weight)#, shuffle=True)
		#training = model.fit_generator(generate_training_data(dataTrain, resultTrain, hyperParams['batchSize']), steps_per_epoch=(len(dataTrain) // hyperParams['batchSize']), epochs=hyperParams['nEpoch'], validation_data = (dataTest, resultTest, sample_weight))
		model.plotResults(dataTest, resultTest, resultScaling, dictMassIndTest, bkgTest, ptphietamOutput, training, outputDir)
		return training.history['val_mean_squared_error'][-1]
	except KeyboardInterrupt:
		model.plotResults(dataTest, resultTest, resultScaling, dictMassIndTest, bkgTest, ptphietamOutput, outputDir)
		return 10**10

class MassModel(tf.keras.Model):
	def __init__(self, inputShape, nOutput, nLayer=3, nNodes=100, activation="relu", dropout=0.3, batchSize = 25, nConvLayer = 1, kernel = (2, 2), nKernels = 1, nEpoch = 500, LSTMcells = 0):
		self.title = "l" + str(nLayer) + "_n" + str(nNodes) + "_" + str(activation) + "_b" + str(batchSize)
		
		if (nConvLayer > 0):
			self.title += "_lconv" + str(nConvLayer) + "_k" + str(nKernels) + "_" + str(kernel[0]) + "x" + str(kernel[1])
		if (LSTMcells > 0):
			self.title += "_lstm" + str(LSTMcells)
		self.title += "_mu_and_e"
		inputLayer = tf.keras.layers.Input(inputShape)	
		#LSTM input layers
		if (LSTMcells > 0):
			x = tf.keras.layers.LSTM(LSTMcells, return_sequences = True)(inputLayer)
		else:
			x = inputLayer
		#Convolutional layers
		for n in range(nConvLayer):
			x = tf.keras.layers.Conv2D(nKernels, kernel, padding = 'same', activation = activation)(x)
		#Dense layers
		flattenLayer = tf.keras.layers.Flatten()(x)
		x = flattenLayer
		x = tf.keras.layers.Dense(nNodes, activation=activation)(x)
		for n in range(nLayer - 1):
			x = tf.keras.layers.Dense(nNodes, activation=activation)(x)
			x = tf.keras.layers.Dropout(rate = dropout)(x)
		#Output layer
		outputLayer = tf.keras.layers.Dense(nOutput)(x)
		tf.keras.Model.__init__(self, inputs=inputLayer, outputs=outputLayer, name="MassModel")


	def plotResults(self, dataTest, resultTest, resultScaling, dictMassIndTest, bkgTest, ptphietamOutput, training=False, outputDir = ""):
		self.summary()
		os.makedirs("models/" + outputDir + self.title, exist_ok=True)
		path_to_save = "models/" + outputDir + self.title + "/" + self.title
		self.save_weights(path_to_save, save_format='tf')

		if (bkgTest is False):
			bkgTest = dataTest[dictMassIndTest[0][0]:dictMassIndTest[0][1]]
		bkgPrediction = self.predict(bkgTest)
		bkgPrediction = bkgPrediction * resultScaling[1] + resultScaling[0]

		if (training):
			self.plotTrainingCurve(training, outputDir="")

		for mass in dictMassIndTest:
			start = dictMassIndTest[mass][0]
			end = dictMassIndTest[mass][1]
			signalPrediction = self.predict(dataTest[start:end])
			signalPrediction = signalPrediction * resultScaling[1] + resultScaling[0]
			self.plotOutputs(signalPrediction, bkgPrediction, resultTest[start:end], mass, outputDir)
			if (ptphietamOutput == False):
				self.plotDerivedMass(signalPrediction, bkgPrediction, resultTest[start:end], mass)
			self.plotCorrelations(signalPrediction, resultTest[start:end], mass, outputDir)

	def plotTrainingCurve(self, training, outputDir=""):
		fig, ax = plt.subplots()
		x_length = len(training.history['mean_squared_error'])
		ax.plot(range(x_length), training.history['mean_squared_error'], label = "Training set")
		ax.plot(range(x_length), training.history['val_mean_squared_error'], label = "Validation set")
		ax.set_ylabel('mean squared error')
		ax.set_xlabel('Number of epochs')
		ax.set_title("Mean squared error of training/validation sample")
		ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

		fig.subplots_adjust(wspace=0.3, hspace=0.7)
		training_name = "models/" + outputDir + self.title + "/" + "training_" + self.title + ".pdf"	
		fig.savefig(training_name, bbox_inches="tight")
		plt.close(fig)

	def plotDerivedMass(self, signalPrediction, bkgPrediction, true, mass):
		mHpredict = np.sqrt(signalPrediction[:, 0]**2 - signalPrediction[:, 1]**2 - signalPrediction[:, 2]**2 - signalPrediction[:, 3]**2)
		mHtrue = np.sqrt(true[:, 0]**2 - true[:, 1]**2 - true[:, 2]**2 - true[:, 3]**2)
		mHbkg = np.sqrt(bkgPrediction[:, 0]**2 - bkgPrediction[:, 1]**2 - bkgPrediction[:, 2]**2 - bkgPrediction[:, 3]**2)
		outputDict = {'xlabel': r'm(H$^\pm$)' + ', GeV',
						'xlim': (0, 1000),
						'bins': 40,
						'title': r"Mass reconstruction of H$^\pm$ ",
						'title_add': "", 
						'name': "mH"}
		fig, ax = plt.subplots()
		nBins = outputDict['bins']
		bins = np.linspace(outputDict['xlim'][0], outputDict['xlim'][1], nBins)
		ax.set_xlim(outputDict['xlim'][0], outputDict['xlim'][1])
		ax.hist(mHtrue, bins = bins, label = "True", histtype="step", normed=True)	
		ax.hist(mHpredict, bins = bins, label = "Prediction (signal)", histtype="step", normed=True)
		ax.hist(mHbkg, bins = bins, label = "Prediction (background)", histtype="step", normed=True)
		ax.set_ylabel("Number of events")
		ax.set_xlabel(outputDict['xlabel'])
		ax.set_yscale("log", nonposy='clip')
		outputDict['title_add'] = r"(m$_{gen}(H)$ = " + str(mass) + " GeV)" 
		histTitle = outputDict['title'] + outputDict['title_add']
		format_string = r"{}"
		ax.set_title(format_string.format(histTitle))
		ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

		fig.subplots_adjust(wspace=0.3, hspace=0.7)
		mass_name = "models/" + outputDir + self.title + "/" + "mass" + str(mass) + "_" + self.title + "_" + outputDict['name'] + ".pdf"
		fig.savefig(mass_name, bbox_inches="tight")
		plt.close(fig)		

	def plotOutputs(self, signal, background, true, mass, outputDir=""):
		for outputNum in range(self.nOutput):
			fig, ax = plt.subplots()
			outputDict = self.OutputDescription[outputNum]
			nBins = outputDict['bins']
			bins = np.linspace(outputDict['xlim'][0], outputDict['xlim'][1], nBins)
			ax.hist(true[:, outputNum], bins = bins, label = "True", histtype="step", normed=True)
			ax.hist(signal[:, outputNum], bins = bins, label = "Prediction (signal)", histtype="step", normed=True)
			ax.hist(background[:, outputNum], bins = bins, label = "Prediction (background)", histtype="step", normed=True)
			ax.set_ylabel("Normalized bin count", fontsize=12)
			ax.set_xlabel(outputDict['xlabel'], fontsize=12)
			ax.set_xlim(outputDict['xlim'][0], outputDict['xlim'][1])
			ax.set_yscale("log", nonposy='clip')
			outputDict['title_add'] = r"(m$_{gen}(H)$ = " + str(mass) + " GeV)" 
			histTitle = outputDict['title'] + outputDict['title_add']
			format_string = r"{}"
			ax.set_title(format_string.format(histTitle), fontsize=12)
			ax.legend(loc = 3, bbox_to_anchor=(-0.15, -0.25, 1., -0.35), fontsize=12, ncol = 3)
			fig.subplots_adjust(wspace=0.3, hspace=0.7)
			mass_name = "models/" + outputDir + self.title + "/" + "mass" + str(mass) + "_" + self.title + "_" + outputDict['name'] + ".pdf"
			fig.savefig(mass_name, bbox_inches="tight")
			plt.close(fig)

	def plotCorrelations(self, signal, true, mass, outputDir):
		for outputNum in range(self.nOutput):
			fig, ax = plt.subplots()
			outputDict = self.OutputDescription[outputNum]
			nBins = 25 # outputDict['bins']
			ax.hist2d(true[:, outputNum], signal[:, outputNum], bins=nBins, norm=mpl.colors.LogNorm())
			outputDict['title_add'] = r"(m$_{gen}(H)$ = " + str(mass) + " GeV)"
			histTitle = "Correlation plot for " + outputDict['xlabel'] + " " + r" (m$_{gen}(H)$ = " + str(mass) + " GeV)"
			ax.set_title(histTitle)
			ax.set_xlabel(outputDict['xlabel'] + " true")
			ax.set_ylabel(outputDict['xlabel'] + " predicted")

			fig.subplots_adjust(wspace=0.3, hspace=0.7)			
			correlation_name = "models/" + outputDir + self.title + "/" + "mass" + str(mass) + "_" + self.title + "_" + outputDict['name'] + "_corr" + ".pdf"
			fig.savefig(correlation_name, bbox_inches="tight")
			plt.close(fig)
