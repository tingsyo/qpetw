#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Encode the QPESUMS data with pre-trained Convolutional Neural Network.
- This version is based on TensorFlow 2.0
- QPESUMS input data format: numpy array with shape (275, 162, 6)
"""
import os, csv, logging, argparse,h5py, json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as cfm
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.utils import normalize, to_categorical

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.8.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-10-01'

# Parameters
nLayer = 6                      # 6 10-min dbz for an hour
nY = 275                        # y-dimension of dbz data
nX = 162                        # x-dimension of dbz data
prec_bins=[0, 0.5, 10, 15, 30, 1000]
yseg_stat = [0.5, 8, 13, 29]    # 40-year statistics of hourly precipitation of trace, 90%, 95%, and 99% percentile
yseg = [0.5, 10, 15, 30]        # Actual segmentation for precipitation

#-----------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------
# Load input/output data for model
def loadInput(srcx):
    # Scan for raw input
    logging.info("Reading input X from: "+ srcx)
    xfiles = []
    for root, dirs, files in os.walk(srcx): 
        for fn in files: 
            if fn.endswith('.npy'): 
                 xfiles.append({'date':fn.replace('.npy',''), 'xuri':os.path.join(root, fn)})
    xfiles = pd.DataFrame(xfiles)
    logging.info("... read input size: "+str(xfiles.shape))
    # Done
    return(xfiles)

def encode_qpesums(encoder, finfo, outdir):
    import numpy as np
    import os
    logging.info("Encoding data with pre-trained autoencoder on "+str(finfo.shape[0])+" files.")
    # Prepare output
    if not os.path.exists(outdir):
        print('Output directory ['+outdir+'] does not exist, create it.')
        os.makedirs(outdir)
    # Perform encoding    
    for i in range(finfo.shape[0]):
        X = np.load(finfo['xuri'].iloc[i])                  # Load QPESUMS data
        xenc = encoder.predict(np.expand_dims(X, axis=0))   # Encode QPESUMS data
        ofile = os.path.join(outdir, finfo['date'].iloc[i]+'.enc.npy')
        np.save(ofile, xenc)                                # Save file
    return(0)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--encoder', '-e', help='the h5 file contains the pre-trained encoder model.')
    parser.add_argument('--output', '-o', help='the file to store training history.')
    parser.add_argument('--logfile', '-l', default='reg.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    #-------------------------------
    # Load raw QPESUMS data information
    finfo = loadInput(args.rawx)
    # Load pre-trained model
    encoder = load_model(args.encoder)
    logging.debug(model[0].summary())
    # Encode data with encoder
    encode_qpesums(encoder, finfo, args.output)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
