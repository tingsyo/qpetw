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

#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
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
#-----------------------------------------------------------------------
def main():
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
    # Set up logging
    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO, filename='evcwb.log', filemode='w')
    # Load data
    qpe = pd.read_csv('cwbqpe_eval.csv')
    qpe['date'] = qpe['timestamp'].apply(lambda x: int(x/100))
    y = pd.read_csv('../examples/data/t1hr.csv')
    # Loop through stations
    report_reg = []
    report_bc = []
    for sid in stdids:
        print(sid)
        tmp = qpe.loc[:,['date',sid]].merge(y.loc[:,['date',sid]], suffixes=('_qpe','_obs'), on='date')
        # Retrieve 2016 data
        idx2016 = np.floor(tmp['date'].astype(int)/1000000) == 2016
        size_of_2016 = sum(idx2016)
        logging.info('    Data index before 2016: '+str(size_of_2016))
        tmp = tmp.loc[idx2016,:].dropna().reset_index(drop=True)
        logging.info('    Data shape: '+str(tmp.shape))
        # remove negative QPE
        tmp[sid+'_qpe'].loc[(tmp[sid+'_qpe']<=0.)] = 0.
        #print(tmp.head())
        # Evaluate
        evreg = evaluate_regression(tmp[sid+'_obs'], tmp[sid+'_qpe'], stid=sid)
        report_reg.append(evreg)
        logging.info(evreg)
        evbc = evaluate_binary(tmp[sid+'_obs'], tmp[sid+'_qpe'], stid=sid)
        report_bc.append(evbc)
        logging.info(evbc)
        # Making plot
        plot_regression(tmp[sid+'_obs'], tmp[sid+'_qpe'], output_prefix='evplot/'+str(sid))
    # Output results
    pd.DataFrame(report_reg).to_csv('evcwbqpe_reg.csv', index=False)
    logging.info(pd.DataFrame(report_reg).describe())
    pd.DataFrame(report_bc).to_csv('evcwbqpe_bc.csv', index=False)
    logging.info(pd.DataFrame(report_bc).describe())
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
