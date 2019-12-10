#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network based Regression for Quantitative Precipitation Estimation.
- Read in QPESUMS data and precipitation data
- Initialize the CNN model
- Train and test
- Output the model
This version is based on TensorFlow 2.0
"""
import os, csv, logging, argparse, glob, h5py, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as cfm
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.utils import normalize, to_categorical

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.3.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-10-27'

# Parameters
nLayer = 6                      # 6 10-min dbz for an hour
nY = 275                        # y-dimension of dbz data
nX = 162                        # x-dimension of dbz data

#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
# Function to give report
def report_prediction(y_pred, timestamp):
    import sklearn.metrics as metrics
    results = []
    for y in y_pred:
        yc = onehot_to_category((y>0.5)*1)
        tmp = {'date': timestamp, 
               'y0':y[0], 
               'y1':y[1], 
               'y2':y[2], 
               'y3':y[3], 
               'y4':y[4],
               'prediction':yc>=1}
        results.append(tmp)
    # Return results
    return(pd.DataFrame(results))

def onehot_to_category(y):
    '''Convert one-hot encoding back to category'''
    return(np.max(np.where(y==1)))

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Perfrom QPE with preprocessed QPESUMS data and pre-trained model.')
    parser.add_argument('--input', '-i', help='the file containing preprocessed QPESUMS data.')
    parser.add_argument('--output', '-o', help='the file to store training history.')
    parser.add_argument('--model_file', '-m', default=None, help='prefix of the pre-trained model files.')
    parser.add_argument('--logfile', '-l', default='reg.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    #-------------------------------
    # Load input data
    #-------------------------------
    logging.info('Loading input data from '+args.input)
    x = np.load(args.input)
    #-------------------------------
    # Load pre-trained model
    #-------------------------------
    logging.info('Loading pre-trained model from '+args.model_file)
    with open(args.model_file+'_model.json') as f:
        model = model_from_json(f.read())
    model.load_weights(args.model_file+'_weights.h5')
    #model.summary()
    #-------------------------------
    # Perform prediction
    #-------------------------------
    logging.info('Performing prediction ')
    y_pred = model.predict(np.expand_dims(x, axis=0), verbose=1)
    print(y_pred)
    y = report_prediction(y_pred, args.input.replace('.npy',''))
    print(y)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
