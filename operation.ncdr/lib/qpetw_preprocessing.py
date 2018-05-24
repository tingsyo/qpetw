#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script of using Keras to implement a 1D convolutional neural network (CNN) for regression.
"""
import os, csv, logging, argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.externals import joblib

def search_dbz(srcdir):
    fileinfo = []
    results = []
    for subdir, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.endswith('.txt'):
                # Parse file name for time information
                furi = os.path.join(subdir, f)
                finfo = f.split('.')
                #logging.debug([furi] + finfo[1:3])
                fileinfo.append([furi] + finfo[1:3])
    return(fileinfo)

def read_dbz(furi):
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2]))
    except pd.errors.EmptyDataError:
        logging.warning(furi + " is empty.")
    return(results)


def transform_dbz(ipca, finfo):
    dbz = []
    # Loop through finfo
    for i in range(0,len(finfo)):
        f = finfo[i]
        logging.debug('Reading data from: ' + f[0])
        tmp = read_dbz(f[0])
        # Append new record
        if tmp is None:     # Copy the previous record if new record is empty
            print('File empty: '+f[0])
            dbz.append(np.zeros(ipca.n_components))
        else:
            tmp = tmp.reshape(1,len(tmp))
            tmp = ipca.transform(tmp).flatten()
            dbz.append(tmp)
    # Save changes of the storage file
    print(np.array(dbz).shape)
    return(dbz)


def writeToCsv(output, fname, header=None):
    # Overwrite the output file:
    with open(fname, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_ALL)
        if header is not None:
            writer.writerow(header)
        for r in output:
            writer.writerow(r)
    return(0)

#-----------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing all the DBZ data.')
    parser.add_argument('--output', '-o', default='output.csv', help='the output file.')
    parser.add_argument('--model', '-m', default='model/dbz.pca20.pca.mod', help='the PCA model file.')
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    print('Sample size: '+str(len(finfo)))

    # Fit Incremental PCA
    logging.info("Transform data with pre-trained model stored in "+ str(args.model))
    ipca = joblib.load(args.model)
    evr = np.cumsum(ipca.explained_variance_ratio_)
    logging.info("Model info: explained variance ratio "+ str(evr))
    logging.info("Model info: transformation dimension "+ str(ipca.components_.shape))
    # Transform data
    dbz_ipca = transform_dbz(ipca, finfo)
    # Append date and projections
    proj_header = ['date','hhmm'] + ['pc'+str(x+1) for x in range(20)]
    newrecs = []
    for i in range(len(finfo)):
        newrecs.append(finfo[i][1:3] + list(dbz_ipca[i]))
    # Output
    writeToCsv(newrecs, args.output, header=proj_header)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()




