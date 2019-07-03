#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binary classification of precipitation (40mm/hr) with the QPESUMS data volume 
"""
import os, csv, logging, argparse, glob, h5py, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
# For cross-validation
from sklearn.model_selection import StratifiedKFold, train_test_split
# For Model
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers import Embedding
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import initializers
from keras.utils import normalize, to_categorical
import keras.backend as K

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019~, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.0.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-07-01'

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
    xdate = [f.split('.')[0].split(os.path.sep)[-1] for f in x]
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
    md = pd.DataFrame({'date':ydate, 'y':np.array(y.loc[:,ylab], dtype=np.float32), 'xuri':None})
    md = md.iloc[xidx,:]
    md.loc[:,'xuri'] = xuri
    # Scan for complete x-y records
    cd = md.loc[~np.isnan(md['y']),:]
    cd.loc[:,'ycat'] = np.digitize(cd['y'], yseg)
    # Done
    return(cd)

# Load input/output data for model
def loadIOTab(srcx, srcy, dropna=False):
    import pandas as pd
    # Read raw input and output
    #logging.info("Reading input X from: "+ srcx)
    print("Reading input X from: "+ srcx)
    xfiles = []
    for root, dirs, files in os.walk(srcx): 
        for fn in files: 
            if fn.endswith('.npy'): 
                 xfiles.append({'date':fn.replace('.npy',''), 'xuri':os.path.join(root, fn)})
    xfiles = pd.DataFrame(xfiles)
    print("... read input size: "+str(xfiles.shape))
    #logging.info("Reading output Y from: "+ srcy)
    print("Reading output Y from: "+ srcy)
    yraw = pd.read_csv(srcy, encoding='utf-8')
    yraw['date'] = yraw['date'].apply(str)
    print("... read output size: "+str(yraw.shape))
    # Create complete IO-data
    print("Pairing X-Y and splitting training/testing data.")
    iotab = pd.merge(yraw, xfiles, on='date', sort=True)
    print("... data size after merging: "+str(iotab.shape))
    # Dro NA if specified
    if dropna:
        print('Dropping records with NA')
        iotab = iotab.dropna()
        print("... data size after dropping-NAs: "+str(iotab.shape))
    # Done
    return(iotab)

# Create cross validation splits
def create_CV_splits(iotable, k=5, ysegs=None, ylab='t1hr', shuffle=False):
    from sklearn.model_selection import StratifiedKFold, KFold
    # Index of each fold
    cvidx_train = []
    cvidx_test = []
    # If segmentation of y is not specified, use simple KFold
    if ysegs is None:
        kf = KFold(n_splits=k, random_state=None, shuffle=shuffle)
        for idx_train, idx_test in kf.split(iotable['xuri']):
            cvidx_train.append(idx_train)
            cvidx_test.append(idx_test)
    else:
        kf = StratifiedKFold(n_splits=k, random_state=None, shuffle=shuffle)
        for idx_train, idx_test in kf.split(iotable['xuri'], np.digitize(iotable[ylab], ysegs)):
            cvidx_train.append(idx_train)
            cvidx_test.append(idx_test)
    return((cvidx_train, cvidx_test))

def to_onehot(y, nclasses=5):
    ''' Represent the given y vector into one-hot encoding of 5 classes (0,1,2,3,4). '''
    L = len(y)                                          # Length of vector y
    yoh = np.zeros((L, nclasses), dtype=np.float32)     # The one-hot encoding, initialized with 0
    for i in range(L):
        yoh[i, 0:y[i]] = 1                              # Encode the corresponding class
        yoh[i, y[i]] = 1                                # Encode the corresponding class
    return(yoh)

# VGG model
def init_model(input_shape, extra_input_shape):
    """
    :Return: 
      a Keras Model to predict y using a 2D vector.
    :param 
      tuple-of-int input_shape: The dimension of main input data volume.
      int extra_input_shape: The number of extra variables as input features.
    #
    Model Design:
      main_input -> convolutional network -> encoded_input --> dense_network_block -> main_output
                                                            /
      extra_input ------------------------------------------ 
    """
    # Convolutional Sub-network
    # Input layer
    main_inputs = Input(shape=input_shape)
    # blovk1: CONV -> CONV -> MaxPooling
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='block1_conv1', data_format='channels_first')(main_inputs)
    x = MaxPooling2D((2,2), name='block1_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(64, (3,3), activation='relu', name='block2_conv1',data_format='channels_first')(x)
    x = MaxPooling2D((2,2), name='block2_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block3: CONV -> CONV -> MaxPooling
    x = Conv2D(128, (3,3), activation='relu', name='block3_conv1',data_format='channels_first')(x)
    x = MaxPooling2D((2,2), name='block3_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block4: CONV -> CONV -> MaxPooling
    x = Conv2D(256, (3,3), activation='relu', name='block4_conv1',data_format='channels_first')(x)
    x = MaxPooling2D((2,2), name='block4_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # Output block: Flatten -> Dense -> Dense -> softmax output
    encoded_input = Flatten()(x)
    # End of Convolutional Sub-network
    # Extra input
    extra_inputs = Input(shape=(extra_input_shape,))
    # Final Output Network
    merged = keras.layers.Add()([encoded_input, extra_input])
    fon = Dense(256, activation='relu', name='fc1')(merged)
    fon = Dropout(0.8)(fon)
    fon = Dense(128, activation='relu', name='fc2')(fon)
    # Output layer
    out = Dense(1, activation='sigmoid', name='final_output')(fon)
    # Initialize model
    model = Model(inputs = [main_inputs, exyta_inputs], outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01, clipvalue=1.)
    #sgd = SGD(lr=0.1, momentum=1e-8, decay=0.01, nesterov=True, clipvalue=1.)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return(model)

def loadDBZ(flist):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        xdata.append(tmp)
    x = np.array(xdata, dtype=np.float32)
    return(x)
    
def data_generator(iotab, batch_size):
    ''' Data generator for batched processing. '''
    nSample = len(iotab)
    y = np.array(iotab['ycat'])
    x = np.array(iotab['xuri'])
    #print(y[:5])
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(x[batch_start:limit])
            Y = to_onehot(y[batch_start:limit])
            #print(X.shape)
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size
    # End of generator
    
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--rawy', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--output', '-o', help='the file to store training history.')
    parser.add_argument('--split', '-s', default=0.2, type=np.float32, help='testing split ratio.')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size.')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    parser.add_argument('--log', '-l', default='mlc.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    #-------------------------------
    # IO data generation
    #-------------------------------
    iotab = loadIOTab(args.rawx, args.rawy, test_split=args.split, shuffle=True)
    #-------------------------------
    # Test dnn
    #-------------------------------
    # Train
    #logging.info("Training model with " + str(len(x)) + " samples.")
    model = init_model((nLayer, nY, nX))
    # Debug info
    print(model.summary())
    print("Training data samples: "+str(iotab['train'].shape[0]))
    steps_train = int(len(iotab['train'])/args.batch_size) + 1
    print("Training data steps: " + str(steps_train))
    print(iotab['train'][:5])
    print("Testing data samples: "+ str(iotab['test'].shape[0]))
    steps_test = int(len(iotab['test'])/args.batch_size) + 1
    print("Testing data steps: " + str(steps_test))
    print(iotab['test'][:5])
    # Fitting model
    hist = model.fit_generator(data_generator(iotab['train'], args.batch_size), steps_per_epoch=steps_train, epochs=args.epochs, max_queue_size=args.batch_size, verbose=0)
    # Prediction
    y_pred = model.predict_generator(data_generator(iotab['test'], args.batch_size), steps=steps_test, verbose=0)
    print(y_pred[:5])
    # Prepare output
    yt = iotab['test']['y']
    y_com = pd.DataFrame({'y': yt, 'y_true':iotab['test']['ycat'],'y_pred':[np.argmax(y) for y in y_pred]})
    conftab = pd.crosstab(y_com['y_true'], y_com['y_pred'])
    print(conftab)
    # Output results
    y_com.to_csv('mlc.ys.csv')
    pd.DataFrame(y_pred).to_csv('mlc.preds.csv')
    pd.DataFrame(hist.history).to_csv('mlc.hist.csv')
    # Save model
    model.save(args.output)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
