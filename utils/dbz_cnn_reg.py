#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script read in grid RADAR data and learn with VGG
"""
import os, csv, logging, argparse, glob, h5py, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
# For cross-validation
from sklearn.model_selection import StratifiedKFold, train_test_split
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
def loadIOTab(srcx, srcy, test_split=0.0, shuffle=False):
    # Read raw input and output
    logging.info("Reading input X from: "+ srcx)
    xfiles = glob.glob(srcx+ os.path.sep +'*.npy')
    logging.info("Reading output Y from: "+ srcy)
    yraw = pd.read_csv(srcy)
    # Create complete IO-data
    iotab = createIOTable(xfiles, yraw)   
    nSample = len(iotab)
    # Create data split
    # nTrain = int(nSample*(1-test_split))
    # if shuffle:
    #    iotab = iotab.sample(frac=1).reset_index(drop=True)
    # Create training/testing split using sklearn.model_selection.train_test_split
    iotrain, iotest = train_test_split(iotab, test_size=test_split, stratify=iotab.loc[:,'ycat'], shuffle=shuffle)
    # Done
    return({'train':iotrain, 'test':iotest})

def to_onehot(y, nclasses=5):
    ''' Represent the given y vector into one-hot encoding of 5 classes (0,1,2,3,4). '''
    L = len(y)                                          # Length of vector y
    yoh = np.zeros((L, nclasses), dtype=np.float32)     # The one-hot encoding, initialized with 0
    for i in range(L):
        yoh[i, y[i]] = 1                                # Encode the corresponding class
    return(yoh)

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
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='block1_conv1', data_format='channels_first', kernel_initializer=initializers.glorot_normal())(inputs)
    x = Conv2D(32, (3,3), activation='relu', name='block1_conv2', data_format='channels_first',kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(32, (3,3), activation='relu', name='block1_conv3', data_format='channels_first',kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), name='block1_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(64, (3,3), activation='relu', name='block2_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(64, (3,3), activation='relu', name='block2_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(64, (3,3), activation='relu', name='block2_conv3',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), name='block2_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block3: CONV -> CONV -> MaxPooling
    x = Conv2D(128, (3,3), activation='relu', name='block3_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(128, (3,3), activation='relu', name='block3_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = Conv2D(128, (3,3), activation='relu', name='block3_conv3',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((2,2), name='block3_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # Output block: Flatten -> Dense -> Dense -> softmax output
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.8)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    # Output layer
    out = Dense(1, activation='linear', name='main_output')(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01, clipvalue=1.)
    #sgd = SGD(lr=0.1, momentum=1e-8, decay=0.01, nesterov=True, clipvalue=1.)
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])
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
    y = np.array(iotab['ycat']).reshape(nSample, 1)
    #print(y[:5])
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(iotab.loc[batch_start:limit,'xuri'])
            Y = y[batch_start:limit]
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
    parser.add_argument('--log', '-l', default='reg.log', help='the log file.')
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
    hist = model.fit_generator(data_generator(iotab['train'], args.batch_size), steps_per_epoch=steps_train,
           epochs=1, use_multiprocessing=True, verbose=1)
    # Prediction
    y_pred = model.predict_generator(data_generator(iotab['test'], args.batch_size), steps=steps_test,
             use_multiprocessing=True, verbose=1)
    print(y_pred[:5])
    # Prepare output
    yt = iotab['test']['y']
    y_com = pd.DataFrame({'y': yt, 'y_true':iotab['test']['ycat'],'y_pred':[np.argmax(y) for y in y_pred]})
    conftab = pd.crosstab(y_com['y_true'], y_com['y_pred'])
    print(conftab)
    # Output results
    with open('reg.ys.csv','w') as cf:
        y_com.to_csv(cf)
    #
    with open('reg.preds.csv','w') as cf2:
        pd.DataFrame(y_pred).to_csv(cf2)
    #
    with open(args.output, 'wb') as fout:
        pickle.dump({'iotable': iotab, 'history':hist.history, 'y':y_com, 'pred':y_pred}, fout)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
