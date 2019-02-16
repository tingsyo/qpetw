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
    

def fit_ipca_partial(finfo, nc=20, bs=100):
    ipca = IncrementalPCA(n_components=nc, batch_size=bs)
    # Loop through finfo
    for i in range(0, len(finfo), bs):
        # Read batch data
        dbz = []
        i2 = i + bs
        if i2>len(finfo):
            i2 = len(finfo)
        for j in range(i, i2):
            f = finfo[j]
            logging.debug('Reading data from: ' + f[0])
            tmp = read_dbz(f[0])
            # Append new record
            if tmp is not None:     # Append data if it is not None
                dbz.append(tmp)
        # Partial fit with batch data
        dbz = np.array(dbz)
        print(dbz.shape)
        ipca.partial_fit(dbz)
    #
    return(ipca)


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
    parser.add_argument('--n_components', '-n', default=20, type=int, help='number of component to output.')
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='number of component to output.')
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    print('Sample size: '+str(len(finfo)))

    # Fit Incremental PCA
    logging.info("Performing IncrementalPCA with "+ str(args.n_components)+" components and batch size of" + str(args.batch_size))
    ipca = fit_ipca_partial(finfo, nc=args.n_components, bs=args.batch_size)
    ev = ipca.explained_variance_
    evr = ipca.explained_variance_ratio_
    com = np.transpose(ipca.components_)
    logging.info("Explained variance ratio: "+ str(evr))
    # Output components
    com_header = ['pc'+str(x+1) for x in range(args.n_components)]
    writeToCsv(com, args.output.replace('.csv','.components.csv'), header=com_header)
    pd.DataFrame({'ev':ev, 'evr':evr}).to_csv(args.output.replace('.csv','.exp_var.csv'))
    # Output fitted IPCA model
    joblib.dump(ipca, args.output.replace(".csv",".pca.mod"))

    # Transform data
    dbz_ipca = transform_dbz(ipca, finfo)
    # Append date and projections
    proj_header = ['date','hhmm'] + ['pc'+str(x+1) for x in range(args.n_components)]
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




