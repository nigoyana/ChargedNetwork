import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import sklearn

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
	#diff_sqrt = tf.math.sqrt(diff)
	diff_sqrt_div = tf.math.divide(diff, tf.square(y_true))
	return tf.math.exp(diff_sqrt_div)
	#return diff

def custom_loss_4vector(y_true, y_pred):
	massTrueSq = y_true[:, 0]**2 - y_true[:, 1]**2 - y_true[:, 2]**2 - y_true[:, 3]**2
	massPredSq = y_pred[:, 0]**2 - y_pred[:, 1]**2 - y_pred[:, 2]**2 - y_pred[:, 3]**2
	return np.sum((massTrueSq - massPredSq)**2)

folder = "test/"

def tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassIndTrain, dictMassIndTest, bkgTest, hyperParams):
	
	nOutput = 4

	model = MassModel(dataTrain.shape[-1], nOutput, **hyperParams)

	model.nOutput = nOutput
	##Change OutputNames!
	model.OutputDescription = [{'xlabel': r'E(H$^\pm$)',
								'xlim': (0, 4000),
								'bins': 400,
								'title': r"Energy reconstruction of H$^\pm$ ",
								'title_add': "", 
								'name': "EH"},
								{'xlabel': r'p$_x$(H$^\pm$)',
								'xlim': (-1000, 1000),
								'bins': 200,
								'title': r"p$_x$ reconstruction for H$^\pm$ ",
								'title_add': "", 
								'name': "pxH"},
								{'xlabel': r'p$_y$(H$^\pm$)',
								'xlim': (-1000, 1000),
								'bins': 200,
								'title': r"p$_y$ reconstruction for H$^\pm$ ",
								'title_add': "", 
								'name': "pyH"},
								{'xlabel': r'p$_z$(H$^\pm$)',
								'xlim': (-2500, 2500),
								'bins': 500,
								'title': r"p$_z$ reconstruction for H$^\pm$ ",
								'title_add': "", 
								'name': "pzH"},

								{'xlabel': r'$\eta$(H$^\pm$)',
								'xlim': (-7., 7.),
								'bins': 140,
								'title': r"$\eta$ reconstruction for H$^\pm$ ",
								'title_add': "", 
								'name': "etaH"},
								{'xlabel': r'$\phi$(H$^\pm$)',
								'xlim': (-4,4),
								'bins': 80,
								'title': r"$\phi$ reconstruction for H$^\pm$ ",
								'title_add': "", 
								'name': "phiH"},
								{'xlabel': r'p$_t$(H$^\pm$)',
								'xlim': (0, 1000),
								'bins': 100,
								'title': r"p$_t$ reconstruction for H$^\pm$ ",
								'title_add': "", 
								'name': "ptH"},

								{'xlabel': r'm(h$_1$)',
								'xlim': (0, 200),
								'bins': 80,
								'title': r"Mass reconstruction of h$_1$ ",
								'title_add': "",
								'name': "h1"},			
								{'xlabel': r'm(h$_2$)',
								'xlim': (0, 200),
								'title': r"Mass reconstruction of h$_2$ ",	
								'title_add': "",
								'name': "h2"}]

	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
	#loss_weights = np.array(getLossWeights(dictMassIndTrain))
	#model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=custom_loss_4vector, metrics=['mean_squared_error']) ##Learning rate -> HyperParams
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

	training = model.fit(dataTrain, resultTrain, epochs=hyperParams['nEpoch'], batch_size=hyperParams['batchSize'], callbacks=[callback], validation_split=0.1, verbose=2, sample_weight = sample_weight)

	model.summary()
	os.makedirs("models/" + folder + model.title, exist_ok=True)
	path_to_save = "models/" + folder + model.title + "/" + model.title
	model.save_weights(path_to_save, save_format='tf')
	##Check mass distribution on test data
	bkgPrediction = model.predict(bkgTest)
	for mass in dictMassIndTest:
		start = dictMassIndTest[mass][0]
		end = dictMassIndTest[mass][1]
		signalPrediction = model.predict(dataTest[start:end])
		model.plotOutputs(signalPrediction, bkgPrediction, resultTest[start:end], mass)
		model.plotDerivedMass(signalPrediction, bkgPrediction, resultTest[start:end], mass)
	model.plotTrainingCurve(training)
	return training.history['val_mean_squared_error'][-1]

class MassModel(tf.keras.Model):
	def __init__(self, inputShape, nOutput, nLayer=3, nNodes=100, activation="relu", dropout=0.3, batchSize = 25, nEpoch = 500):
		
		self.title = "l" + str(nLayer) + "_n" + str(nNodes) + "_" + str(activation) + "_b" + str(batchSize) + "_mu_and_e"
		
		inputLayer = tf.keras.layers.Input(shape=(inputShape))
		x = tf.keras.layers.Dense(nNodes, activation=activation)(inputLayer)

		for n in range(nLayer-1):
			x = tf.keras.layers.Dense(nNodes, activation=activation)(x)

		outputLayer = tf.keras.layers.Dense(nOutput)(x)

		tf.keras.Model.__init__(self, inputs=inputLayer, outputs=outputLayer, name="MassModel")

	def plotTrainingCurve(self, training):
		fig, ax = plt.subplots()
		x_length = len(training.history['mean_squared_error'])
		ax.plot(range(x_length), training.history['mean_squared_error'], label = "Training set")
		ax.plot(range(x_length), training.history['val_mean_squared_error'], label = "Validation set")
		ax.set_ylabel('mean squared error')
		ax.set_xlabel('Number of epochs')
		ax.set_title("Mean squared error of training/validation sample")
		ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

		fig.subplots_adjust(wspace=0.3, hspace=0.7)
		training_name = "models/" + folder + self.title + "/" + "training_" + self.title + ".pdf"	
		fig.savefig(training_name, bbox_inches="tight")

	def plotDerivedMass(self, signalPrediction, bkgPrediction, true, mass):
		mHpredict = np.sqrt(signalPrediction[:, 0]**2 - signalPrediction[:, 1]**2 - signalPrediction[:, 2]**2 - signalPrediction[:, 3]**2)
		mHtrue = np.sqrt(true[:, 0]**2 - true[:, 1]**2 - true[:, 2]**2 - true[:, 3]**2)
		mHbkg = np.sqrt(bkgPrediction[:, 0]**2 - bkgPrediction[:, 1]**2 - bkgPrediction[:, 2]**2 - bkgPrediction[:, 3]**2)
		outputDict = {'xlabel': r'm(H$^\pm$)',
								'xlim': (0, 800),
								'bins': 80,
								'title': r"Mass reconstruction of H$^\pm$ ",
								'title_add': "", 
								'name': "mH"}
		fig, ax = plt.subplots()
		nBins = outputDict['bins']
		ax.hist(mHtrue, bins=nBins, label = "True", histtype="step",normed=True)	
		ax.hist(mHpredict, bins=nBins, label = "Prediction (signal)", histtype="step", normed=True)
		ax.hist(mHbkg, bins=nBins, label = "Prediction (background)", histtype="step", normed=True)
		ax.set_ylabel("Number of events")
		ax.set_xlabel(outputDict['xlabel'])
		ax.set_xlim(outputDict['xlim'][0], outputDict['xlim'][1])
		ax.set_yscale("log", nonposy='clip')
		outputDict['title_add'] = r"(m$_{gen}(H)$ = " + str(mass) + " GeV)" 
		histTitle = outputDict['title'] + outputDict['title_add']
		format_string = r"{}"
		ax.set_title(format_string.format(histTitle))
		ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

		fig.subplots_adjust(wspace=0.3, hspace=0.7)
		mass_name = "models/" + folder + self.title + "/" + "mass" + str(mass) + "_" + self.title + "_" + outputDict['name'] + ".pdf"
		fig.savefig(mass_name, bbox_inches="tight")			

	def plotOutputs(self, signal, background, true, mass):
		for outputNum in range(self.nOutput):
			fig, ax = plt.subplots()
			outputDict = self.OutputDescription[outputNum]
			nBins = outputDict['bins']
			ax.hist(true[:, outputNum], bins=nBins, label = "True", histtype="step",normed=True)
			ax.hist(signal[:, outputNum], bins=nBins, label = "Prediction (signal)", histtype="step", normed=True)
			ax.hist(background[:, outputNum], bins=nBins, label = "Prediction (background)", histtype="step", normed=True)
			ax.set_ylabel("Number of events")
			ax.set_xlabel(outputDict['xlabel'])
			ax.set_xlim(outputDict['xlim'][0], outputDict['xlim'][1])
			ax.set_yscale("log", nonposy='clip')
			outputDict['title_add'] = r"(m$_{gen}(H)$ = " + str(mass) + " GeV)" 
			histTitle = outputDict['title'] + outputDict['title_add']
			format_string = r"{}"
			ax.set_title(format_string.format(histTitle))
			ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
			fig.subplots_adjust(wspace=0.3, hspace=0.7)
			mass_name = "models/" + folder + self.title + "/" + "mass" + str(mass) + "_" + self.title + "_" + outputDict['name'] + ".pdf"
			fig.savefig(mass_name, bbox_inches="tight")
			#mass_name = "models/" + folder + self.title + "/" + "mass" + str(mass) + "_" + self.title + "_" +  outputDict['name'] + ".png"
			#fig.savefig(mass_name, bbox_inches="tight")
			

