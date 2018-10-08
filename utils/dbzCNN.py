#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script read in grid RADAR data and learn with VGG
"""
import os, csv, logging, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
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
nChannel = 6                    # 6 10-min dbz for an hour
nX = 275                        # x-dimension of dbz data
nY = 162                        # y-dimension of dbz data
nSample = 37369                 # Total complete samples
yseg_stat = [0.5, 8, 13, 29]    # 40-year statistics of hourly precipitation of trace, 90%, 95%, and 99% percentile
yseg = [0.5, 10, 15, 30]        # Actual segmentation for precipitation

# VGG model
def init_model_vgg(input_shape):
    """
    :Return: 
      a Keras VGG-like Model to predict y using a 2D vector (regression).
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    # blovk1: CONV -> CONV -> MaxPooling
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', name='block1_conv1', kernel_initializer=initializers.glorot_normal())(inputs)
    x = Conv2D(64, (3,3), activation='relu', name='block1_conv2', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), strides=(1,1), name='block1_pool')(x)
    x = Dropout(0.5)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(128, (3,3), activation='relu', name='block2_conv1', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(128, (3,3), activation='relu', name='block2_conv2', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), strides=(1,1), name='block2_pool')(x)
    x = Dropout(0.5)(x)
    # block3: CONV -> CONV -> MaxPooling
    x = Conv2D(256, (3,3), activation='relu', name='block3_conv1', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(256, (3,3), activation='relu', name='block3_conv2', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(256, (3,3), activation='relu', name='block3_conv3', kernel_initializer=initializers.glorot_normal())(x)
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
    #adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01, clipvalue=1.)
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
    parser.add_argument('--input', '-i', help='the directory containing all the DBZ data.')
    parser.add_argument('--output', '-o', default='output.csv', help='the output file.')
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='batch size.')
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Read input and output
    logging.info("Extract data from all files: "+ str(len(finfo)))

    logging.info("Extract data from all files: "+ str(len(finfo)))
    
    # Set up cross validation
    logging.info("Extract data from all files: "+ str(len(finfo)))
    
    # Train and Test
    logging.info("Performing IncrementalPCA with "+ str(args.n_components)+" components.")
    
    logging.debug("Explained variance ratio: "+ str(evr))
    # Output results
    
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
