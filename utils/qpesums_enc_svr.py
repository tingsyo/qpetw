#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group evaluation for Linear Regression with NN-Encoded QPESUMS.
- Read in encoded QPESUMS data
- Read in precipitation data of 45 stations
- Loop through 45 stations
  - train with 2013~2015 and evaluate on 2016
"""
import sys, os, csv, logging, argparse, h5py
import numpy as np
import pandas as pd
from sklearn import linear_model, svm

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019~2020, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-12-20'

# Parameters
stdids = ['466880', '466910', '466920', '466930', '466940', 
          'C0A520', 'C0A530', 'C0A540', 'C0A550', 'C0A560', 
          'C0A570', 'C0A580', 'C0A640', 'C0A650', 'C0A660', 
          'C0A710', 'C0A860', 'C0A870', 'C0A880', 'C0A890', 
          'C0A920', 'C0A940', 'C0A950', 'C0A970', 'C0A980', 
          'C0A9A0', 'C0A9B0', 'C0A9C0', 'C0A9E0', 'C0A9F0', 
          'C0A9G0', 'C0A9I1', 'C0AC40', 'C0AC60', 'C0AC70', 
          'C0AC80', 'C0ACA0', 'C0AD00', 'C0AD10', 'C0AD20', 
          'C0AD30', 'C0AD40', 'C0AD50', 'C0AG90', 'C0AH00']
#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
# Load input/output data for model
def loadIOTab(srcx, srcy, dropna=False):
    import pandas as pd
    import os
    # Read raw input and output
    #logging.info("Reading input X from: "+ srcx)
    logging.info("Reading input X from: "+ srcx)
    xfiles = []
    for root, dirs, files in os.walk(srcx): 
        for fn in files: 
            if fn.endswith('.npy'): 
                 xfiles.append({'date':fn.replace('.enc.npy',''), 'xuri':os.path.join(root, fn)})
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

# Function to give report for binary classifications
def evaluate_binary(yt, yp, stid=None, ythresh=30.):
    from sklearn.metrics import confusion_matrix
    ytb = (yt>=ythresh)*1
    ypb = (yp>=ythresh)*1
    # Derive metrics
    output = {'id':stid}
    TN, FP, FN, TP = confusion_matrix(ytb, ypb).ravel()
    output['true_positive'] = np.round(TP,2)
    output['false_positive'] = np.round(FP,2)
    output['false_negative'] = np.round(FN,2)
    output['true_negative'] = np.round(TN,2)
    output['sensitivity'] = np.round(TP/(TP+FN),2)
    output['specificity'] = np.round(TN/(FP+TN),2)
    output['prevalence'] = np.round((TP+FN)/(FN+TP+FP+TN),8)
    output['ppv'] = np.round(TP/(TP+FP),4)
    output['npv'] = np.round(TN/(TN+FN),4)
    output['fpr'] = np.round(FP/(FP+TN),4)
    output['fnr'] = np.round(FN/(FN+TP),4)
    output['fdr'] = np.round(FP/(FP+TP),4)
    output['FOR'] = np.round(FN/(TN+FN),4)
    output['accuracy'] = np.round((TP+TN)/(FN+TP+FP+TN),4)
    output['F1'] = np.round(2*TP/(2*TP+FP+FN),4)
    output['MCC'] = np.round((TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),4)
    output['informedness'] = np.round(output['sensitivity'] + output['specificity'] - 1,4)
    output['markedness'] = np.round(output['ppv'] + output['npv'] -1,4)
    return(output)

# Function to give report for regression
def evaluate_regression(y_true, y_pred, stid=None):
    import sklearn.metrics as metrics
    # Calculate measures
    results = {'id':stid}
    results['y_true_mean'] = y_true.mean()
    results['y_true_var'] = y_true.var()
    results['y_pred_mean'] = y_pred.mean()
    results['y_pred_var'] = y_pred.var()
    results['rmse'] = np.sqrt(metrics.mean_squared_error(y_true,y_pred))
    if y_pred.var()<=10e-8:
        results['corr'] = 0
    else:
        results['corr'] = np.corrcoef(y_true,y_pred)[0,1]
    # Return results
    return(results)

# Plot evaluation of regression
def plot_regression(y_true, y_pred, output_prefix=None):
    import matplotlib.pyplot as plt
    # Show time series
    plt.subplot(2,1,1)
    plt.plot(y_true, label='true')
    plt.plot(y_pred, '--r', label='pred')
    plt.ylim(0,80)
    plt.title('Time series')
    plt.legend()
    # Show scatter plot
    plt.subplot(2,1,2)
    plt.scatter(y_pred, y_true)
    plt.xlim(0,80)
    plt.ylim(0,80)
    plt.title('Predictions vs Truth')
    plt.tight_layout()
    # Save to file if specified
    if not output_prefix is None:
        plt.savefig(output_prefix+'.png')
        plt.close()
        return(0)
    else:
        return(plt)

def y_to_log(y):
    ''' Convert the y to log(y+1). '''
    ylog = np.log(y+1).astype(np.float32)
    return(ylog)

def log_to_y(y):
    ''' Convert the predicted y in log-scale back to original scale. '''
    yori = (np.exp(y.flatten())-1.0).astype(np.float32)
    yori[yori<0.5] = 0.                          # Set the minimal values to 0.
    return(yori)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--xpath', '-x', help='the directory containing ebcoded QPESUMS data.')
    parser.add_argument('--ypath', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--yth', '-t', default=-1, type=float, help='Threshold of Y for training.')
    parser.add_argument('--logy', '-g', default=0, type=int, choices=range(0, 2), help='Use Y in log-space.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # IO data generation
    iotab = loadIOTab(args.xpath, args.ypath, dropna=False)
    logging.info('    number of total records listed: '+str(iotab.shape[0]))
    # Load Input
    x_full=[]
    for i in range(iotab.shape[0]):
        x_full.append(np.load(iotab['xuri'].iloc[i]).flatten())
    x_full = pd.DataFrame(np.array(x_full))
    x_full.index = list(iotab['date'])
    logging.info('    number of total records read: '+str(x_full.shape[0]))
    # Loop through stations
    report_train = []
    report_test = []
    for sid in stdids:
        # Create iotable for the station
        logging.info('Station id: '+sid)
        stdio = iotab.loc[:,['date', sid]].merge(x_full, left_on='date', right_index=True).dropna().reset_index(drop=True)
        logging.info('    number of valid records: '+str(stdio.shape[0]))
        y = stdio[sid]
        x = stdio.iloc[:, 2:]
        # Split training and testing data
        idx2016 = np.floor(stdio['date'].astype(float)/1000000.) == 2016
        size_of_2016 = sum(idx2016)
        size_before_2016 = sum(stdio['date'].astype(int)<2016010100)
        logging.info('    Data index of 2016: '+str(size_of_2016))
        #y_train = y.iloc[:size_before_2016]
        #x_train = x.iloc[:size_before_2016,:]
        y_train = y.loc[~idx2016]
        x_train = x.loc[~idx2016,:]
        # Apply filter on training data
        idx_filtered = y_train>args.yth
        logging.info('    Filter y: '+str(args.yth))
        y_train = y_train.loc[idx_filtered]
        x_train = x_train.loc[idx_filtered,:]
        # Reporting training/testing size
        logging.info('    Data dimension of training data: '+str(x_train.shape[0]) + ', ' +str(x_train.shape[1]))
        y_test = y.loc[idx2016].reset_index(drop=True)
        x_test = x.loc[idx2016,:].reset_index(drop=True)
        logging.info('    Data dimension of testing data: '+str(x_test.shape[0]) + ', ' +str(x_test.shape[1]))
        # Train model and test
        #reg = linear_model.SGDRegressor(loss='squared_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.25)
        reg = svm.SVR(kernel='poly', degree=2, gamma='scale', coef0=0.0, tol=0.0001, C=0.05, epsilon=0.25)
        #reg = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=4)
        #reg = linear_model.BayesianRidge(normalize=True)
        reg.fit(x_train, y_to_log(y_train))
        yp_train = reg.predict(x_train)
        yp_test = reg.predict(x_test)
        # Evaluate
        evtrain = evaluate_regression(y_train, log_to_y(yp_train), stid=sid)
        report_train.append(evtrain)
        logging.info(evtrain)
        evtest = evaluate_regression(y_test, log_to_y(yp_test), stid=sid)
        report_test.append(evtest)
        logging.info(evtest)
        # Making plot
        plot_regression(y_test, log_to_y(yp_test), output_prefix=('svr_'+str(sid)))
    # Output results
    pd.DataFrame(report_train).to_csv(args.output+'_train.csv', index=False)
    logging.info(pd.DataFrame(report_test).describe())
    pd.DataFrame(report_test).to_csv(args.output+'_test.csv', index=False)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
