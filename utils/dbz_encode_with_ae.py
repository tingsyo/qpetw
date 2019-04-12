#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads RADAR-dbz data in text format (275*162 values with lon/lat), and encode the data
with pre-trained convolutional autoencoder to reduce the data diemnsion. 
"""
import os, csv, logging, argparse, pickle, h5py, json, glob
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2017~2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.0.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-01-09'

# Data dimension
nLayer = 6
nY = 275
nX = 162

# Utilities
def loadDBZ(flist):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        xdata.append(tmp)
    x = np.array(xdata, dtype=np.float32)
    return(x)

def data_generator_ae(flist, batch_size):
    ''' Data generator for batched processing. '''
    nSample = len(flist)
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(flist[batch_start:limit])/100.
            #print(X.shape)
            yield (X,X) #a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size
    # End of generator

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--output', '-o', default='encoded', help='the directory to output the encoded data.')
    parser.add_argument('--encoder', '-e', default='output_encoder.h5', help='the pre-trained encoder model.')
    parser.add_argument('--log', '-l', default='encode_ae.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    #logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    #-------------------------------
    # Find all files
    #-------------------------------
    fs = []
    for root, dirs, files in os.walk(args.rawx):
        for file in files:
            if glob.fnmatch.fnmatch(file, '*.npy'):
                fs.append(file)
    nSample = len(fs)
    #logging.info("Training model with " + str(len(x)) + " samples.")
    #-------------------------------
    # Load pre-trained model
    #-------------------------------
    encoder = load_model(args.encoder)
    # Debug info
    print(encoder.summary())
    print("Encoding data with pre-trained autoencoder on "+str(len(fs))+" files.")
    # Prepare output
    if not os.path.exists(args.output):
        print('Output directory ['+args.output+'] does not exist, create it.')
        os.makedirs(args.output)
    # Encode data
    for f in fs:
        X = np.load(os.path.join(args.rawx,f))/100.         # Scale X with 1/100 to have values 0. ~ 1.
        xenc = encoder.predict(np.expand_dims(X, axis=0))
        ofile = os.path.join(args.output, f)
        np.save(ofile, xenc)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()


