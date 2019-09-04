#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose: Perform Incremental PCA on QPESUMS data.
Description:
--input: directory that contains QPESUMS data as *.npy (6*275*162)
--output: the prefix of output files.
--model: the file contains the trained IncrementalPCA object.
--n_components: number of component to output.
"""
import os, csv, logging, argparse, pickle, h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
import joblib

''' Input processing '''
# Scan QPESUMS data in *.npy: 6*275*162 
def search_dbz(srcdir):
    import pandas as pd
    fileinfo = []
    for subdir, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.endswith('.npy'):
                # Parse file name for time information
                furi = os.path.join(subdir, f)
                finfo = f.split('.')
                ftime = finfo[0]
                #logging.debug([furi] + finfo[1:3])
                fileinfo.append([furi, ftime])
    results = pd.DataFrame(fileinfo, columns=['furi', 'timestamp'])
    results = results.sort_values(by=['timestamp']).reset_index(drop=True)
    return(results)

# Read uris containing QPESUMS data in the format of 6*275*162 
def loadDBZ(flist):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        # Append new record
        if tmp is not None:            # Append the flattened data array if it is not None
            xdata.append(tmp.flatten())
    x = np.array(xdata, dtype=np.float32)
    return(x)

''' Project data into PCs '''
def transform_dbz(ipca, finfo):
    dbz = []
    # Loop through finfo
    for i in range(0,finfo.shape[0]):
        f = finfo.iloc[i,:]
        #logging.debug('Reading data from: ' + f['furi'])
        tmp = np.load(f['furi']).flatten()
        # Append new record
        if tmp is None:     # Copy the previous record if new record is empty
            logging.warning('File empty: '+f['furi'])
            dbz.append(np.zeros(ipca.n_components))
        else:
            tmp = tmp.reshape(1,len(tmp))
            tmp = ipca.transform(tmp).flatten()
            dbz.append(tmp)
    # Save changes of the storage file
    logging.debug('Data dimension after projection: ' + str(np.array(dbz).shape))
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
    parser.add_argument('--output', '-o', default='output', help='the output file.')
    parser.add_argument('--model', '-m', help='the trained IncrementalPCA object stored with joblib.')
    parser.add_argument('--n_components', '-n', default=20, type=int, help='number of component to output.')
    parser.add_argument('--log', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.log is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.log, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    logging.debug('Total data size: '+str(finfo.shape[0]))
    # Load Incremental PCA model
    logging.info("Load IncrementalPCA model from " + args.model)
    ipca = joblib.load(args.model)
    # Summarize results
    nc = ipca.n_components
    ev = ipca.explained_variance_
    evr = ipca.explained_variance_ratio_
    com = np.transpose(ipca.components_)
    logging.info("Model summary: ")
    logging.info("    Number of components: "+ str(nc))
    logging.info("    Explained variance ratio: "+ str(evr))
    # Transform the data with loaded model
    dbz_ipca = transform_dbz(ipca, finfo)
    # Append date and projections
    proj_header = ['timestamp'] + ['pc'+str(x+1) for x in range(nc)]
    newrecs = []
    for i in range(finfo.shape[0]):
        newrecs.append([finfo['timestamp'].iloc[i]] + list(dbz_ipca[i]))
    # Output
    writeToCsv(newrecs, args.output+".csv", header=proj_header)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
