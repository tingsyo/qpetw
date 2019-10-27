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
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
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
batchSize = 128                 # Batch size for training / testing

#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
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
    # Generate weited sampling

    # Done
    return(iotab)

def generate_samples(iotab, ylab='y', prec_bins=[0, 1, 5, 10, 20, 40, 500], num_epoch=100, shuffle=True):
    '''Create weighted sampling list'''
    # Analysis the Precipitation
    prec_hist = np.histogram(iotab[ylab], bins=prec_bins)
    p = 1/(prec_hist[0]/prec_hist[0][-1])           # Calculate probability
    p = p/sum(p)                                    # Normalize the probability
    n = sum(prec_hist[0])*num_epoch                 # Total number of samples
    nrep = np.round(n*p/prec_hist[0]).astype(int)   # Convert to numbers of sampling
    # Categorize precipitation by specified bins
    iotab['prec_cat'] = np.digitize(iotab[ylab], bins=prec_bins)
    print(iotab['prec_cat'].value_counts())
    # Repeat sampling by p
    for icat in range(1,len(prec_bins)):
        repeat_n = nrep[icat-1]
        tmp = iotab.loc[iotab['prec_cat']==icat,:]
        print('Append data category: '+str(icat)+' for '+ str(repeat_n) +' times with size '+str(tmp.shape))
        for j in range(int(repeat_n)):
            iotab = iotab.append(tmp, ignore_index=True)
    # Shuffle new dataset if specified
    if shuffle:
        iotab = iotab.sample(frac=1)#.reset_index(drop=True)
    #
    return(iotab)

# CNN
def init_model_reg(input_shape):
    """
    :Return: 
      Newly initialized model (regression).
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    # blovk1: CONV -> CONV -> MaxPooling
    x = Conv2D(filters=8, kernel_size=(3,3), activation='relu', name='block1_conv1', data_format='channels_first')(inputs)
    x = MaxPooling2D((2,2), name='block1_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(16, (3,3), activation='relu', name='block2_conv1',data_format='channels_first')(x)
    x = MaxPooling2D((2,2), name='block2_pool', data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    # Output block: Flatten -> Dense -> Dense -> softmax output
    x = Flatten()(x)
    x = Dense(8, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='relu', name='fc2')(x)
    # Output layer
    out = Dense(1, activation='linear', name='main_output')(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #sgd = SGD(lr=0.01, momentum=1e-8, decay=0.001, nesterov=True)#, clipvalue=1.)
    model.compile(loss='mse', optimizer=adam, metrics=['mae','cosine_similarity'])
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

def data_generator_reg(iotab, batch_size, ylab='y', logy=False):
    ''' Data generator for batched processing. '''
    nSample = len(iotab)
    y = np.array(iotab[ylab], dtype=np.float32).reshape(nSample, 1)
    #print(y[:5])
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(iotab['xuri'][batch_start:limit])/100.
            if logy:
                Y = y_to_log(y[batch_start:limit])
            else:
                Y = y[batch_start:limit]
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

# Create cross validation splits
def create_CV_splits(iotable, k=5, ysegs=None, ylab='y', shuffle=False):
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
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--rawy', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--output', '-o', help='the file to store training history.')
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='number of epochs.')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    parser.add_argument('--kfold', '-k', default=2, type=int, help='number of folds for cross validation.')
    parser.add_argument('--log', '-l', default='reg.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    #logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    #-------------------------------
    # IO data generation
    #-------------------------------
    iotab = loadIOTab(args.rawx, args.rawy, dropna=True)
    #-------------------------------
    # Create weited sampling rom IOdata
    #-------------------------------
    #iotab = pd.DataFrame({'date':iotab.date, 't1hr':iotab.t1hr})
    print(iotab.head())
    iotab = generate_samples(iotab, ylab='t1hr', prec_bins=[0, 1, 5, 10, 20, 40, 500], num_epoch=10, shuffle=False)
    print(iotab.head())
    #-------------------------------
    # Create Cross Validation splits
    #-------------------------------
    idx_trains, idx_tests = create_CV_splits(iotab, k=args.kfold, ysegs=[0.5, 5, 10], ylab='t1hr', shuffle=False)
    #-------------------------------
    # Run through CV
    #-------------------------------
    ys = []
    hists = []
    cv_report = []
    for i in range(len(idx_trains)):
        # Train
        #logging.info("Training model with " + str(len(x)) + " samples.")
        model = init_model_reg((nLayer, nY, nX))
        # Debug info
        if i==0:
            print(model.summary())
        print("Training data samples: "+str(len(idx_trains[i])))
        steps_train = np.ceil(len(idx_trains[i])/args.batch_size)
        print("Training data steps: " + str(steps_train))
        print("Testing data samples: "+ str(len(idx_tests[i])))
        steps_test = np.ceil(len(idx_tests[i])/args.batch_size)
        print("Testing data steps: " + str(steps_test))
        # Fitting model
        hist = model.fit_generator(data_generator_reg(iotab.iloc[idx_trains[i],:], args.batch_size, ylab='t1hr'), steps_per_epoch=steps_train, epochs=args.epochs, max_queue_size=args.batch_size, verbose=0)
        # Prediction
        y_pred = model.predict_generator(data_generator_reg(iotab.iloc[idx_tests[i],:], args.batch_size, ylab='t1hr'), steps=steps_test, verbose=0)
        # Prepare output
        yt = np.array(iotab['t1hr'])[idx_tests[i]]
        ys.append(pd.DataFrame({'y': yt, 'y_pred': y_pred.flatten()}))
        hists.append(hist.history) 
        cv_report.append(report_evaluation(yt, y_pred))
        # Debug info
        print('Histogram of y_true: ')
        print(np.histogram(yt, bins=[0, 1, 5, 10, 40, 1000]))
        #print(np.histogram(y_to_log(yt)))
        print('Histogram of y_pred: ')
        print(np.histogram(y_pred, bins=[0, 1, 5, 10, 40, 1000]))
        #print(np.histogram(y_pred))
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
