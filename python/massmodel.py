import matplotlib.pyplot as plt
import tensorflow as tf
import os

def tryModel(dataTrain, resultTrain, dataTest, resultTest, dictMassInd, bkgTest, hyperParams):
	model = MassModel(dataTrain.shape[-1], 1, **hyperParams)
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
	model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
	training = model.fit(dataTrain, resultTrain, epochs=hyperParams['nEpoch'], batch_size=hyperParams['batchSize'], callbacks=[callback], validation_split=0.1, verbose=2)
	model.summary()
	os.makedirs("models/random_search/" + model.title, exist_ok=True)
	path_to_save = "models/random_search/" + model.title + "/" + model.title
	model.save_weights(path_to_save, save_format='tf')
	##Check mass distribution on test data
	bkgPrediction = model.predict(bkgTest).flatten()
	for mass in dictMassInd:
		start = dictMassInd[mass][0]
		end = dictMassInd[mass][1]
		signalPrediction = model.predict(dataTest[start:end]).flatten()
		model.plotTraining(training, signalPrediction, bkgPrediction, resultTest[start:end], mass)
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

	def plotTraining(self, training, signal, background, true, mass):
		fig, ax = plt.subplots()
		x_length = len(training.history['mean_squared_error'])
		ax.plot(range(x_length), training.history['mean_squared_error'], label = "Training set")
		ax.plot(range(x_length), training.history['val_mean_squared_error'], label = "Validation set")
		ax.set_ylabel('mean squared error')
		ax.set_xlabel('Number of epochs')
		ax.set_title("Mean squared error of training/validation sample")
		ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

		fig.subplots_adjust(wspace=0.3, hspace=0.7)
		training_name = "models/random_search/" + self.title + "/" + "training_" + self.title + ".pdf"
		fig.savefig(training_name, bbox_inches="tight")

		fig, ax = plt.subplots()
		ax.hist(true, bins=80, label = "True", histtype="step",normed=True)
		ax.hist(signal, bins=80, label = "Prediction (signal)", histtype="step", normed=True)
		ax.hist(background, bins=80, label = "Prediction (background)", histtype="step", normed=True)
		ax.set_ylabel("Number of events")
		ax.set_xlabel(r'm(H$^\pm$)')
		ax.set_xlim(0, 800)
		histTitle = r"Mass reconstruction of H$^\pm$ with m = " + str(mass) +" GeV"
		ax.set_title(histTitle)
		ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

		fig.subplots_adjust(wspace=0.3, hspace=0.7)
		mass_name = "models/random_search/" + self.title + "/" + "mass" + str(mass) + "_" + self.title + ".pdf"
		fig.savefig(mass_name, bbox_inches="tight")

