#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script read in grid RADAR data and learn with VGG
"""
import os, csv, logging, argparse, glob, h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
# For cross-validation
from sklearn.model_selection import StratifiedKFold
# For fixing random state: block1
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(14221)
rn.seed(12345)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# For fixing random state: block1
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers import Embedding
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import initializers
from keras.utils import normalize, to_categorical
import keras.backend as K
# For fixing random state: block2
tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
# For fixing random state: block2

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2017~2018, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.0.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2018-02-14'

# Parameters
nSample = 37369                 # Total complete samples
nLayer = 6                      # 6 10-min dbz for an hour
nY = 275                        # y-dimension of dbz data
nX = 162                        # x-dimension of dbz data
batchSize = 128                 # Batch size for training / testing
yseg_stat = [0.5, 8, 13, 29]    # 40-year statistics of hourly precipitation of trace, 90%, 95%, and 99% percentile
yseg = [0.5, 10, 15, 30]        # Actual segmentation for precipitation

#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
def createIOTable(x, y, ylab='t1hr', qpf=False):
    ''' 
    Given the input/output data, check for data completeness and match records by date.
    - Start with ydate because the precipitation date is always complete.
    - Loop through x, associate x with y of the same date.
    - Check the flag QPF. If QPF, shift the x(t)-y(t) mapping to y(t)-x(t-1).
    - Check through x-y list, keep only complete cases.
    - Return input/output.
    '''
    import pandas as pd
    # Clean up dates of x and y
    xdate = [f.split('.')[0].split('/')[-1] for f in x]
    ydate = np.array(y['date'], dtype='str')
    xfull = zip(xdate, x)                                   # Pair up date of x and the uri
    x_in_y = [x for x in xfull if x[0] in ydate]            # Valid xdate: xdate existing in ydate
    xd, xuri = zip(*x_in_y)
    xidx = [list(ydate).index(x) for x in xd]               # Index of valid xdate in ydate
    # Check qpf flag, if true use earlier xdate
    if(qpf):
        xidx = [x+1 for x in xidx if(x+1)<len(ydate)]
        x_in_y = x_in_y[:len(xidx)]
    # Match days
    md = pd.DataFrame({'date':ydate, 'y':np.array(y[ylab], dtype=np.float32), 'x':None})
    md = md.iloc[xidx,:]
    md['x'] = xuri
    # Scan for complete x-y records
    cd = []
    for i in range(md.shape[0]):
        if not (None in list(md.iloc[i,:])):
            cd.append(list(md.iloc[i,:]))
    # Done
    return(pd.DataFrame(cd, columns=['date','y','xuri']))

# Load input/output data for model
def loadIOData(srcx, srcy, xhdf5):
    # Read raw input and output
    logging.info("Reading input X from: "+ srcx)
    xfiles = glob.glob(srcx+'/*.npy')
    logging.info("Reading output Y from: "+ srcy)
    yraw = pd.read_csv(srcy)
    # Create complete IO-data
    iotab = createIOTable(xfiles, yraw)   
    nSample = len(iotab)
    # Create paired Input data
    fx = h5py.File(xhdf5, 'w')
    xdata = fx.create_dataset('x', (nSample, nLayer, nY, nX), chunks=(batchSize, nLayer, nY, nX), dtype=np.float32)
    for i in range(nSample):
        tmp = np.load(iotab['xuri'][i])
        xdata[i] = tmp
    # Create categorized output data
    y = np.array(iotab['y'], dtype=np.float32).reshape(nSample, 1)
    ydata = np.digitize(y, yseg)
    #ydata = to_categorical(ycat)
    # Done
    return(ydata, xdata)

# VGG model
def init_model(input_shape):
    """
    :Return: 
      a Keras VGG-like Model to predict y using a 2D vector (regression).
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    # blovk1: CONV -> CONV -> MaxPooling
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', name='block1_conv1', data_format='channels_first', kernel_initializer=initializers.glorot_normal())(inputs)
    x = Conv2D(64, (3,3), activation='relu', name='block1_conv2', data_format='channels_first',kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), strides=(1,1), name='block1_pool')(x)
    x = Dropout(0.5)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(128, (3,3), activation='relu', name='block2_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(128, (3,3), activation='relu', name='block2_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), strides=(1,1), name='block2_pool')(x)
    x = Dropout(0.5)(x)
    # block3: CONV -> CONV -> MaxPooling
    x = Conv2D(256, (3,3), activation='relu', name='block3_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(256, (3,3), activation='relu', name='block3_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(256, (3,3), activation='relu', name='block3_conv3',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), strides=(1,1), name='block3_pool')(x)
    x = Dropout(0.5)(x)
    # Output block: Flatten -> Dense -> Dense -> softmax output
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.8)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    # Output layer
    out = Dense(5, activation='sigmoid', name='main_output')(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    sgd = SGD(lr=0.1, momentum=1e-8, decay=0.01, nesterov=True, clipvalue=1.)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return(model)

def read_x(furi):
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2])).reshape((275,162))
    except pd.errors.EmptyDataError:
        logging.warning(furi + " is empty.")
    return(results)

def read_y(furi):
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2])).reshape((275,162))
    except pd.errors.EmptyDataError:
        logging.warning(furi + " is empty.")
    return(results)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--rawy', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--input', '-i', default='x.hdf5', help='the processed input data.')
    parser.add_argument('--output', '-o', default='y.hdf5', help='the processed output data.')
    parser.add_argument('--kfold', '-k', default=3, type=int, help='K-fold cross validation.')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size.')
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    #-------------------------------
    # IO data generation
    #-------------------------------
    y, x = loadIOData(args.rawx, args.rawy, args.input)
    print("Data dimension:")
    print(x.shape)
    #-------------------------------
    # Cross validation
    #-------------------------------
    # Set up cross validation
    logging.info("Generate data splits for "+ str(args.kfold)+"-fold cross validation.")
    skf = StratifiedKFold(n_splits=args.kfold)
    H = []
    T = []
    for trainIdx, testIdx in skf.split(x, y):
        # Get indeces of training/testing set
        xTrain, xTest = x[trainIdx,:,:,:], x[testIdx,:,:,:]
        yTrain, yTest = to_categorical(y[trainIdx]), to_categorical(y[testIdx])
        # Train
        logging.info("Training model with " + str(len(trainIdx)) + " samples.")
        model = init_model((xTrain.shape[1:]))
        hist = model.fit(xTrain, yTrain, epochs=10, batch_size=128, initial_epoch=0, verbose=1)
        # Test
        logging.info("Testing model with " + str(len(testIdx)) + " samples.")
        test = model.evaluate(xTest, yTest, batch_size=128)
        # Output results
        H.append(hist)
        T.append(test)
    #
    for test in T:
        print(test)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
