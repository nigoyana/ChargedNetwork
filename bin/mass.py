#!/usr/bin/env python3

from massmodel import MassModel

import argparse
import yaml
import os

import numpy as np
import pandas as pd
import tensorflow as tf

def parser():
    parser = argparse.ArgumentParser(description='Program for predict charged Higgs mass', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action = "store_true", help="Train model on data")

    return parser.parse_args()

def trainModel():
    data = []
    result = []

    nEpoch = 200

    ##Read csv data to numpy
    dataFrame = pd.read_csv("data/massTraining/L4B_200_100.csv", sep=",")
    bkgFrame = pd.read_csv("data/massTraining/TT+j-1L.csv", sep=",")
    data = dataFrame.to_numpy()

    trainFrac =  int(0.9*data.shape[0])

    dataTest, resulTest = data[trainFrac::, :-12],  data[trainFrac::, [-1]]
    dataTrain, resultTrain = data[:trainFrac:, :-12],  data[:trainFrac:, [-1]] 
    bkgTest = bkgFrame.to_numpy()[:40000:, :-12],

    ##Train model
    model = MassModel(3, 3, 50, "relu", 0.4)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
    training = model.fit(dataTrain, resultTrain, epochs=nEpoch, batch_size=25, validation_split=0.1, verbose=2)
    model.summary()

    ##Check mass distribution on test data
    signalPrediction = model.predict(dataTest).flatten()
    bkgPrediction = model.predict(bkgTest).flatten()
    model.plotTraining(nEpoch, training, signalPrediction, bkgPrediction, resulTest)

def main():
    ##Parser arguments
    args = parser()

    ##Train model
    if args.train:         
          trainModel()  

if __name__ == "__main__":
    main()
