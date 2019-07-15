import matplotlib.pyplot as plt
import tensorflow as tf

class MassModel(tf.keras.Model):
    def __init__(self, nClass, nLayer=3, nNodes=100, activation="relu", dropout=0.3):
        super(MassModel, self).__init__(name="MassModel")
        self.nClass = nClass

        self.inputLayer = tf.keras.layers.Dense(nNodes, activation=activation)
        #self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.layer1 = tf.keras.layers.Dense(nNodes, activation=activation)
        self.outputLayer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        ##Push input through all layers
        x = self.inputLayer(inputs)
        #x = self.dropout1(x)
        x = self.layer1(x)
        return self.outputLayer(x)

    def compute_output_shape(self, inputShape):
        shape = tf.TensorShape(inputShape).as_list()
        shape[-1] = self.nClass
        return tf.TensorShape(shape)

    def plotTraining(self, nEpoch, training, signal, background, true):
        fig, ax = plt.subplots()
        ax.plot(range(nEpoch), training.history['mean_squared_error'], label = "Training set")
        ax.plot(range(nEpoch), training.history['val_mean_squared_error'], label = "Validation set")
        ax.set_ylabel('mean squared error')
        ax.set_xlabel('Number of epochs')
        ax.set_title("Mean squared error of training/validation sample")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

        fig.subplots_adjust(wspace=0.3, hspace=0.7)
        fig.savefig("training.pdf", bbox_inches="tight")

        fig, ax = plt.subplots()
        ax.hist(true, bins=15, label = "True", histtype="step",normed=True)
        ax.hist(signal, bins=15, label = "Prediction (signal)", histtype="step", normed=True)
        ax.hist(background, bins=15, label = "Prediction (background)", histtype="step", normed=True)
        ax.set_ylabel("Number of events")
        ax.set_xlabel(r'm(H$^\pm$)')
        ax.set_title(r"Mass reconstruction of H$^\pm$")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

        fig.subplots_adjust(wspace=0.3, hspace=0.7)
        fig.savefig("mass.pdf", bbox_inches="tight")

