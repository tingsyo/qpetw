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
# For fixing random state: block1
#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(14221)
#rn.seed(12345)
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
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
# For fixing random state: block2

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-06-02'

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
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first', kernel_initializer=initializers.glorot_normal())(inputs)
    #x = Conv2D(32, (3,3), activation='relu', name='block1_conv2', data_format='channels_first',kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(32, (3,3), activation='relu', name='block1_conv3', data_format='channels_first',kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((5,3), name='block1_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(16, (3,3), activation='relu', padding='same', name='block2_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(64, (3,3), activation='relu', name='block2_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(64, (3,3), activation='relu', name='block2_conv3',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((1,3), name='block2_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block3: CONV -> CONV -> MaxPooling
    x = Conv2D(8, (3,3), activation='relu', padding='same', name='block3_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(128, (3,3), activation='relu', name='block3_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(128, (3,3), activation='relu', name='block3_conv3',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((1,3), name='block3_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block4: CONV -> CONV -> MaxPooling
    x = Conv2D(4, (3,3), activation='relu', padding='same', name='block4_conv1',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(256, (3,3), activation='relu', name='block4_conv2',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    #x = Conv2D(256, (3,3), activation='relu', name='block4_conv3',data_format='channels_first', kernel_initializer=initializers.glorot_normal())(x)
    x = MaxPooling2D((5,3), name='block4_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # Output block: Flatten -> Dense -> Dense -> softmax output
    x = Flatten()(x)
    x = Dense(16, activation='relu', name='fc1')(x)
    x = Dropout(0.8)(x)
    x = Dense(8, activation='relu', name='fc2')(x)
    # Output layer
    out = Dense(1, activation='linear', name='main_output')(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01, clipvalue=1.)
    #sgd = SGD(lr=0.01, momentum=1e-8, decay=0.001, nesterov=True)#, clipvalue=1.)
    model.compile(loss='mse', optimizer=adam, metrics=['mae','cosine_proximity'])
    return(model)

def loadDBZ(flist):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        xdata.append(tmp)
    x = np.array(xdata, dtype=np.float32)
    return(x)
    
def y_to_log(y):
    ''' Convert the y to log(y+1). '''
    ylog = np.log(y+1).astype(np.float32)
    return(ylog)

def log_to_y(y):
    ''' Convert the predicted y in log-scale back to original scale. '''
    yori = (np.exp(y.flatten())-1.0).astype(np.float32)
    yori[yori<0.5] = 0.                          # Set the minimal values to 0.
    return(yori)

def data_generator_reg(iotab, batch_size):
    ''' Data generator for batched processing. '''
    nSample = len(iotab)
    y = np.array(iotab['y'], dtype=np.float32).reshape(nSample, 1)
    #print(y[:5])
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(iotab['xuri'][batch_start:limit])/100.
            Y = y_to_log(y[batch_start:limit])
            #print(X.shape)
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size
    # End of generator

# Function to give report
def report_evaluation(y_true, y_pred, verbose=0):
    import sklearn.metrics as metrics
    # Calculate measures
    results = {}
    results['y_true_mean'] = y_true.mean()
    results['y_true_var'] = y_true.var()
    results['y_pred_mean'] = y_pred.mean()
    results['y_pred_var'] = y_pred.var()
    results['rmse'] = np.sqrt(metrics.mean_squared_error(y_true,y_pred))
    if y_pred.var()<=10e-8:
        results['corr'] = 0
    else:
        results['corr'] = np.corrcoef(y_true,y_pred)[0,1]
    # Print results if verbose > 0
    if verbose>0:
        if verbose>1:
            print('Mean of y_true: ' + str(results['y_true_mean']))
            print('Variance of y_true: ' + str(results['y_true_var']))
            print('Mean of y_pred: ' + str(results['y_pred_mean']))
            print('Variance of y_pred: ' + str(results['y_pred_var']))
        print('RMSE: ' + str(results['rmse']))
        print('Corr: ' + str(results['corr']))
    # Return results
    return(results)

    
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--rawy', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--output', '-o', help='the file to store training history.')
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='number of epochs.')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    parser.add_argument('--log', '-l', default='reg.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    #logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    #-------------------------------
    # IO data generation
    #-------------------------------
    iotab = loadIOTab(args.rawx, args.rawy, dropna=True)
    print(iotab.head())
    #-------------------------------
    # Create Cross Validation splits
    #-------------------------------
    idx_trains, idx_tests = create_CV_splits(iotab, k=3, ysegs=[0.5, 5, 10], ylab='t1hr', shuffle=False)
    #-------------------------------
    # Run through CV
    #-------------------------------
    ys = []
    hists = []
    cv_report = []
    iotab_log = pd.DataFrame({'y':y_to_log(iotab['t1hr']), 'xuri':iotab['xuri']})
    for i in range(len(idx_trains)):
        # Train
        #logging.info("Training model with " + str(len(x)) + " samples.")
        model = init_model((nLayer, nY, nX))
        # Debug info
        #print(model.summary())
        print("Training data samples: "+str(len(idx_trains[i])))
        steps_train = np.ceil(len(idx_trains[i])/args.batch_size)
        print("Training data steps: " + str(steps_train))
        print("Testing data samples: "+ str(len(idx_tests[i])))
        steps_test = np.ceil(len(idx_tests[i])/args.batch_size)
        print("Testing data steps: " + str(steps_test))
        # Fitting model
        hist = model.fit_generator(data_generator_reg(iotab_log.iloc[idx_trains[i],:], args.batch_size), steps_per_epoch=steps_train, epochs=args.epochs, max_queue_size=args.batch_size, verbose=0)
        # Prediction
        y_pred = model.predict_generator(data_generator_reg(iotab_log.iloc[idx_tests[i],:], args.batch_size), steps=steps_test, verbose=0)
        yp = log_to_y(y_pred)
        # Debug info
        print('Mean of yp_log: ' + str(y_pred.mean()))
        print('Variance of yp_log: ' + str(y_pred.var()))
        print('Mean of yp: ' + str(yp.mean()))
        print('Variance of yp: ' + str(yp.var()))
        # Prepare output
        yt = np.array(iotab['t1hr'])[idx_tests[i]]
        ys.append(pd.DataFrame({'y': yt, 'y_pred_log': y_pred.flatten(), 'y_pred':yp}))
        hists.append(hist.history) 
        cv_report.append(report_evaluation(yt, yp))
    # Output results
    pd.concat(ys).to_csv(args.output+'_reg_ys.csv')
    pd.DataFrame(hists).to_csv(args.output+'_reg_hist.csv')
    pd.DataFrame(cv_report).to_csv(args.output+'_reg_report.csv')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
