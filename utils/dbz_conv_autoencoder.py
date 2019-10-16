#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads RADAR-dbz data in text format (275*162 values with lon/lat), and performs 
convolutional autoencoder algorithm to reduce the data diemnsion. 
"""
import os, csv, logging, argparse, pickle, h5py, json, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
#from tensorflow.keras import backend as K

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2017~2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.9.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-01-09'

# Data dimension
nLayer = 6
nY = 275
nX = 162

''' Input processing '''
# Scan QPESUMS data in *.npy: 6*275*162 
def search_dbz(srcdir):
    import pandas as pd
    fileinfo = []
    for subdir, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.endswith('.npy'):
                # Parse file name for time information
                furi = os.path.join(subdir, f)
                finfo = f.split('.')
                ftime = finfo[0]
                #logging.debug([furi] + finfo[1:3])
                fileinfo.append([furi, ftime])
    results = pd.DataFrame(fileinfo, columns=['furi', 'timestamp'])
    results = results.sort_values(by=['timestamp']).reset_index(drop=True)
    return(results)

# Read uris containing QPESUMS data in the format of 6*275*162 
def loadDBZ(flist, to_log=False):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        # Append new record
        if tmp is not None:            # Append the flattened data array if it is not None
            xdata.append(tmp)
    x = np.array(xdata, dtype=np.float32)
    # Convert to log space if specified
    if to_log:
        x = np.log(x+1)
    # done
    return(x)

def data_generator_ae(flist, batch_size, to_log=False):
    ''' Data generator for batched processing. '''
    nSample = len(flist)
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(flist[batch_start:limit], to_log)
            #print(X.shape)
            yield (X,X) #a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size
    # End of generator

# Autoencoder model
def initialize_autoencoder_qpesums(input_shape):
    # Define input layer
    input_data = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    # Define encoder layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='encoder_conv1', data_format='channels_first')(input_data)
    x = MaxPooling2D((5, 3), name='encoder_maxpool1', data_format='channels_first')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='encoder_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((1, 3), name='encoder_maxpool2', data_format='channels_first')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder_conv3', data_format='channels_first')(x)
    x = MaxPooling2D((1, 3), name='encoder_maxpool3', data_format='channels_first')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='encoder_conv4', data_format='channels_first')(x)
    encoded = MaxPooling2D((5, 3), name='encoder_maxpool4', data_format='channels_first')(x)
    #encoded = x
    # Define decoder layers
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='decoder_conv1', data_format='channels_first')(encoded)
    x = UpSampling2D((5, 3), name='decoder_upsamp1', data_format='channels_first')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='decoder_conv2', data_format='channels_first')(x)
    x = UpSampling2D((1, 3), name='decoder_upsamp2', data_format='channels_first')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='decoder_conv3', data_format='channels_first')(x)
    x = UpSampling2D((1, 3), name='decoder_upsamp3', data_format='channels_first')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='decoder_conv4', data_format='channels_first')(x)
    x = UpSampling2D((5, 3), name='decoder_upsamp4', data_format='channels_first')(x)
    decoded = Conv2D(6, (3, 3), activation='sigmoid', name='decoder_output', padding='same', data_format='channels_first')(x)
    # Define autoencoder
    autoencoder = Model(input_data, decoded)
    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['cosine_similarity'])
    # Encoder
    encoder = Model(input_data, encoded)
    return((autoencoder, encoder))



#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing all the DBZ data.')
    parser.add_argument('--output', '-o', default='output', help='the output file.')
    parser.add_argument('--filter', '-f', default=None, help='the filter file with time-stamps.')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='size of each data batch.')
    parser.add_argument('--log_flag', '-g', default=1, type=int, choices=range(0, 2), help="convert to log-scale")
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    #-------------------------------
    # Set up logging
    #-------------------------------
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    #-------------------------------
    # Scan files for reading
    #-------------------------------
    finfo = search_dbz(args.input)
    logging.debug('Total data size: '+str(finfo.shape[0]))
    #-------------------------------
    # Apply filter if specified
    #-------------------------------
    if not args.filter is None:
        logging.debug('Read filter file: '+args.filter)
        flt = pd.read_csv(args.filter)
        logging.debug('Filter size: '+str(flt.shape[0]))
        fidx = finfo['timestamp'].isin(list(flt.iloc[:,0].apply(str)))
        finfo = finfo.loc[fidx,:].reset_index(drop=True)
        logging.debug('Data size after filter: '+str(finfo.shape[0]))
    #-------------------------------
    # Test dnn
    #-------------------------------
    # Train
    logging.info("Training model with " + str(finfo.shape[0]) + " samples.")
    ae = initialize_autoencoder_qpesums((nLayer, nY, nX))
    # Debug info
    nSample = finfo.shape[0]
    print(ae[0].summary())
    print("Training autoencoder with data size: "+str(nSample))
    steps_train = np.ceil(nSample/args.batch_size)
    print("Training data steps: " + str(steps_train))
    # Fitting model
    hist = ae[0].fit_generator(data_generator_ae(finfo['furi'], args.batch_size), steps_per_epoch=steps_train,epochs=args.epochs, max_queue_size=args.batch_size, use_multiprocessing=False, verbose=1)
    # Prepare output
    pd.DataFrame(hist.history).to_csv(args.output+'_hist.csv')
    ae[0].save(args.output+'_ae.h5')
    ae[1].save(args.output+'_encoder.h5')
    # done
    return(0)
#==========
# Script
#==========
if __name__=="__main__":
    main()


