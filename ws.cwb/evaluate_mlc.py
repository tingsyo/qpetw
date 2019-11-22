#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network for Quantitative Precipitation Estimation.
- This version is based on TensorFlow 2.0
- QPESUMS input data format: numpy array with shape (275, 162, 6)
"""
import os, csv, logging, argparse,h5py, pickle
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
def loadIOTab(srcx, srcy, dropna=False):
    import pandas as pd
    # Read raw input and output
    #logging.info("Reading input X from: "+ srcx)
    logging.info("Reading input X from: "+ srcx)
    xfiles = []
    for root, dirs, files in os.walk(srcx): 
        for fn in files: 
            if fn.endswith('.npy'): 
                 xfiles.append({'date':fn.replace('.npy',''), 'xuri':os.path.join(root, fn)})
    xfiles = pd.DataFrame(xfiles)
    logging.info("... read input size: "+str(xfiles.shape))
    #logging.info("Reading output Y from: "+ srcy)
    logging.info("Reading output Y from: "+ srcy)
    yraw = pd.read_csv(srcy, encoding='utf-8')
    yraw['date'] = yraw['date'].apply(str)
    logging.info("... read output size: "+str(yraw.shape))
    # Create complete IO-data
    logging.info("Pairing X-Y and splitting training/testing data.")
    iotab = pd.merge(yraw, xfiles, on='date', sort=True)
    logging.info("... data size after merging: "+str(iotab.shape))
    # Dro NA if specified
    if dropna:
        logging.info('Dropping records with NA')
        iotab = iotab.dropna()
        logging.info("... data size after dropping-NAs: "+str(iotab.shape))
    # Done
    return(iotab)

def generate_equal_samples(iotab, prec_bins, ylab='y', shuffle=True):
    '''Create equal sampling list: 
           repeat sample rare categories to mtach the frequency of the majority case.
    '''
    # Analysis the Precipitation
    prec_hist = np.histogram(iotab[ylab], bins=prec_bins)
    maxc = np.max(prec_hist[0])                     # Count the most frequent category
    nrep = np.round(maxc/prec_hist[0]).astype(int)  # Multiples required to reach max-count
    # Categorize precipitation by specified bins
    #iotab['prec_cat'] = np.digitize(iotab[ylab], bins=prec_bins[1:-1])
    logging.debug('Sample histogram before weighted sampling:')
    logging.debug(iotab['prec_cat'].value_counts())
    # Repeat sampling by p
    for icat in range(0,len(prec_bins)-1):
        repeat_n = nrep[icat]
        tmp = iotab.loc[iotab['prec_cat']==icat,:]
        logging.info('Append data category: '+str(icat)+' for '+ str(repeat_n) +' times with size '+str(tmp.shape))
        for j in range(int(repeat_n)-1):
            iotab = iotab.append(tmp, ignore_index=True)
    logging.debug('Sample histogram after weighted sampling:')
    logging.debug(iotab['prec_cat'].value_counts().sort_index())
    # Shuffle new dataset if specified
    if shuffle:
        iotab = iotab.sample(frac=1)#.reset_index(drop=True)
    #
    return(iotab)

def loadDBZ(flist):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        xdata.append(tmp)
    x = np.array(xdata, dtype=np.float32)
    return(x)

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
            logging.info('Mean of y_true: ' + str(results['y_true_mean']))
            logging.info('Variance of y_true: ' + str(results['y_true_var']))
            logging.info('Mean of y_pred: ' + str(results['y_pred_mean']))
            logging.info('Variance of y_pred: ' + str(results['y_pred_var']))
        logging.info('RMSE: ' + str(results['rmse']))
        logging.info('Corr: ' + str(results['corr']))
    # Return results
    return(results)

def to_onehot(y, nclasses=5):
    ''' Represent the given y vector into one-hot encoding of 5 classes (0,1,2,3,4). '''
    L = len(y)                                          # Length of vector y
    yoh = np.zeros((L, nclasses), dtype=np.float32)     # The one-hot encoding, initialized with 0
    for i in range(L):
        yoh[i, 0:y[i]] = 1                              # Encode the corresponding class
        yoh[i, y[i]] = 1                                # Encode the corresponding class
    return(yoh)

def onehot_to_category(y):
    '''Convert one-hot encoding back to category'''
    c = []
    for r in y:
        c.append(np.max(np.where(r==1)))
    return(np.array(c))

def data_generator_mlc(iotab, batch_size, ylab='y'):
    ''' Data generator for batched processing. '''
    nSample = len(iotab)
    y = np.array(iotab[ylab])
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = loadDBZ(iotab['xuri'][batch_start:limit])
            Y = to_onehot(y[batch_start:limit])
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size
    # End of generator

# CNN
def init_model_mlc(input_shape):
    """
    :Return: 
      Newly initialized model (regression).
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    # blovk1: CONV -> MaxPooling
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='block1_conv1', data_format='channels_last')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2,2), name='block1_pool', data_format='channels_last')(x)
    x = Dropout(0.25)(x)
    # block2: CONV -> CONV -> MaxPooling
    x = Conv2D(64, (3,3), activation='relu', name='block2_conv1', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3,3), activation='relu', name='block2_conv2', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2,2), name='block2_pool', data_format='channels_last')(x)
    x = Dropout(0.25)(x)
    # block3: CONV -> CONV -> MaxPooling
    x = Conv2D(128, (3,3), activation='relu', name='block3_conv1', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3,3), activation='relu', name='block3_conv2', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2,2), name='block3_pool', data_format='channels_last')(x)
    x = Dropout(0.25)(x)
    # block4: CONV -> CONV -> MaxPooling
    x = Conv2D(256, (3,3), activation='relu', name='block4_conv1', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(256, (3,3), activation='relu', name='block4_conv2', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2,2), name='block4_pool', data_format='channels_last')(x)
    x = Dropout(0.25)(x)
    # Output block: Flatten -> Dense -> Dense -> softmax output
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    x = BatchNormalization(axis=-1)(x)
    # Output layer
    out = Dense(5, activation='sigmoid', name='main_output')(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    encoder = Model(inputs = inputs, outputs = x)
    return((model, encoder))

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--rawy', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--output', '-o', help='the file to store training history.')
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='number of epochs.')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    parser.add_argument('--logfile', '-l', default='reg.log', help='the log file.')
    parser.add_argument('--random_seed', '-r', default=None, type=int, help='the random seed.')
    parser.add_argument('--model_file', '-m', default=None, help='pre-trained model file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    #-------------------------------
    # IO data generation
    #-------------------------------
    iotab = loadIOTab(args.rawx, args.rawy, dropna=True)
    # Categorize y
    iotab['prec_cat'] = np.digitize(iotab['t1hr'], bins=prec_bins[1:-1])
    #-------------------------------
    # Set random seed if specified
    #-------------------------------
    if not args.random_seed is None:
        tf.random.set_seed(args.random_seed)
    #-------------------------------
    # Create Training/Testing split
    #-------------------------------
    io201314 = iotab.loc[(iotab['date'].astype(int) < 2015010100),:]
    io2015 = iotab.loc[(iotab['date'].astype(int) >= 2015010100) & (iotab['date'].astype(int) < 2016010100), :]
    io2016 = iotab.loc[(iotab['date'].astype(int) >= 2016010100), :]
    #-------------------------------
    # Train and evaluate
    #-------------------------------
    # Create weighted sampling for training data
    iotrain = io201314
    iotrain = generate_equal_samples(iotrain, prec_bins=prec_bins, ylab='t1hr', shuffle=True)
    iotest = io2015
    # Initialize model
    if not args.model_file is None:
        model = load_model()
    else:
        model, encoder = init_model_mlc((nY, nX, nLayer))
    logging.debug(model.summary())
    # Calculate steps 
    steps_train = np.ceil(iotrain.shape[0]/args.batch_size)
    steps_test = np.ceil(iotest.shape[0]/args.batch_size)
    logging.info("Training data samples: "+str(iotrain.shape[0]))
    logging.info("Testing data samples: "+ str(iotest.shape[0]))
    logging.debug("Training data steps: " + str(steps_train))
    logging.debug("Testing data steps: " + str(steps_test))
    # Fitting model
    hist = model.fit_generator(data_generator_mlc(iotrain, args.batch_size, ylab='prec_cat'), 
                                    steps_per_epoch=steps_train, 
                                    epochs=args.epochs, 
                                    max_queue_size=args.batch_size, 
                                    verbose=1)
    # Prediction
    y_pred = model.predict_generator(data_generator_mlc(iotest, args.batch_size, ylab='prec_cat'), 
                                    steps=steps_test, 
                                    verbose=0)
    # Prepare output
    yt = np.array(iotest['prec_cat'])
    yp = onehot_to_category((y_pred>0.5)*1)
    ys = pd.DataFrame({'date': iotest['date'], 
                        'y': yt, 
                        'y_pred': yp, 
                        'y0':y_pred[:,0], 
                        'y1':y_pred[:,1], 
                        'y2':y_pred[:,2], 
                        'y3':y_pred[:,3], 
                        'y4':y_pred[:,4]})
    hists = pd.DataFrame(hist.history)
    cv_report = report_evaluation(yt, yp)
    # Debug info
    logging.info('Confusion Matrix: ')
    logging.info(cfm(yt, yp))
    # Output results
    ys.to_csv(args.output+'_mlc_ys.csv', index=False)
    hists.to_csv(args.output+'_mlc_hist.csv')
    logging.info(cv_report)
    #cv_report.to_csv(args.output+'_mlc_report.csv')
    # Output model
    model[1].save(args.output+'_encoder.h5')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
