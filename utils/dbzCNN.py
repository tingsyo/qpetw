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


# VGG model
def init_model_vgg(input_shape):
    """
    :Return: 
      a Keras VGG-like Model to predict y using a 2D vector (regression).
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Initialize model
    model = Sequential()
    # blovk1: CONV -> CONV -> MaxPooling
    model.add(Conv2D(filters=64, kernel_size=(3,1), strides=(1,1), activation='relu', name='block1_conv1', kernel_initializer=initializers.glorot_normal(), input_shape=input_shape))
    model.add(Conv2D(64, (3,1), activation='relu', name='block1_conv2', kernel_initializer=initializers.glorot_normal()))
    model.add(MaxPooling2D((2,1), strides=(2,1), name='block1_pool'))
    model.add(Dropout(0.2))
    # block2: CONV -> CONV -> MaxPooling
    model.add(Conv2D(128, (3,1), activation='relu', name='block2_conv1', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(128, (3,1), activation='relu', name='block2_conv2', kernel_initializer=initializers.glorot_normal()))
    model.add(MaxPooling2D((2,1), strides=(2,1), name='block2_pool'))
    model.add(Dropout(0.2))
    # block3: CONV -> CONV -> MaxPooling
    model.add(Conv2D(256, (3,1), activation='relu', name='block3_conv1', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(256, (3,1), activation='relu', name='block3_conv2', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(256, (3,1), activation='relu', name='block3_conv3', kernel_initializer=initializers.glorot_normal()))
    model.add(MaxPooling2D((2,1), strides=(2,1), name='block3_pool'))
    model.add(Dropout(0.2))
    # block4: CONV -> CONV -> MaxPooling
    model.add(Conv2D(512, (3,1), activation='relu', name='block4_conv1', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(512, (3,1), activation='relu', name='block4_conv2', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(512, (3,1), activation='relu', name='block4_conv3', kernel_initializer=initializers.glorot_normal()))
    model.add(MaxPooling2D((2,1), strides=(2,2), name='block4_pool'))
    model.add(Dropout(0.2))
    # block5: CONV -> CONV -> MaxPooling -> Dropout
    model.add(Conv2D(512, (3,1), activation='relu', name='block5_conv1', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(512, (3,1), activation='relu', name='block5_conv2', kernel_initializer=initializers.glorot_normal()))
    model.add(Conv2D(512, (3,1), activation='relu', name='block5_conv3', kernel_initializer=initializers.glorot_normal()))
    model.add(MaxPooling2D((2,1), strides=(2,2), name='block5_pool'))
    model.add(Dropout(0.2))
    # Output block: Flatten -> Dense -> Dense -> regression output
    model.add(Flatten())  
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dense(1, activation='linear', name='prediction')) 
    # Define compile parameters
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01, clipvalue=1.)
    model.compile(loss='cosine_proximity', optimizer=adam, metrics=['mse','mae'])
    return(model)

def read_dbz(furi):
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
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='number of component to output.')
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    # Read into numpy.memmap
    logging.info("Extract data from all files: "+ str(len(finfo)))
    dbz = read_dbz_memmap(finfo)
    # Process dbz data with Incremental PCA
    logging.info("Performing IncrementalPCA with "+ str(args.n_components)+" components.")
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_size)
    dbz_ipca = ipca.fit_transform(dbz)
    evr = ipca.explained_variance_ratio_
    com = np.transpose(ipca.components_)
    logging.debug("Explained variance ratio: "+ str(evr))
    # Output components
    com_header = ['pc'+str(x+1) for x in range(args.n_components)]
    #np.insert(com, 0, com_header, axis=0)
    writeToCsv(com, args.output.replace('.csv','.components.csv'), header=com_header)
    # Append date and projections
    proj_header = ['date','hhmm'] + ['pc'+str(x+1) for x in range(args.n_components)]
    newrecs = []
    for i in range(len(finfo)):
        newrecs.append(finfo[i][1:3] + list(dbz_ipca[i]))
    # Output
    writeToCsv(newrecs, args.output, header=proj_header)
    # Save PCA for later use
    joblib.dump(ipca, args.output.replace(".csv",".pca.mod"))
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
