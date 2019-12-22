#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose: Perform Incremental PCA on QPESUMS data.
Description:
--input: directory that contains QPESUMS data as *.npy (6*275*162)
--output: the prefix of output files.
--filter: the file contains a list of timestamp that filters the input data for processing.
--n_components: number of component to output.
--batch_size: the size of data batch for incremental processing, default=100.
--randomseed: integer as the random seed, default="1234543"
"""
import os, sys, csv, logging, argparse, pickle, h5py
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
def loadDBZ(flist, to_log=False):
    ''' Load a list a dbz files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)
        # Append new record
        if tmp is not None:            # Append the flattened data array if it is not None
            xdata.append(tmp.flatten())
    x = np.array(xdata, dtype=np.float32)
    # Convert to log space if specified
    if to_log:
        x = np.log(x+1)
    # done
    return(x)

''' Perform Incremental PCA '''
def fit_ipca_partial(finfo, nc=20, bs=100, log_flag=False):
    nrec = finfo.shape[0]
    # Initialize the IncrementalPCA object
    ipca = IncrementalPCA(n_components=nc, batch_size=bs)
    # Check whether the last batch size is smaller than n_components
    flag_merge_last_batch = False
    if np.mod(nrec, bs)<nc:
        logging.warning('The last batch is smaller than n_component, merge the last two batches.')
        flag_merge_last_batch = True
    # Setup batch counter
    n_batch = int(np.floor(nrec/bs))
    if not flag_merge_last_batch:
        n_batch = n_batch + 1
    logging.debug('Number of batches: '+str(n_batch))
    # Loop through the first (n_batch-1) batch
    for i in range(n_batch-1):
        # Read batch data
        i1 = i * bs
        i2 = i1 + bs
        # Load batch data
        dbz = loadDBZ(finfo['furi'].iloc[i1:i2], to_log=log_flag)
        logging.debug('Batch dimension: '+ str(dbz.shape))
        # Partial fit with batch data
        ipca.partial_fit(dbz)
    # In case there is only one batch
    if n_batch==1:
        i2 = 0
    # Fit the last batch
    dbz = loadDBZ(finfo['furi'].iloc[i2:nrec], to_log=log_flag)
    logging.debug('Final batch dimension: '+ str(dbz.shape))
    ipca.partial_fit(dbz)
    # done
    return(ipca)

''' Project data into PCs '''
def transform_dbz(ipca, finfo, to_log):
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
            # Convert to log space if specified
            if to_log:
                tmp = np.log(tmp+1)
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
    parser.add_argument('--filter', '-f', default=None, help='the filter file with time-stamps.')
    parser.add_argument('--n_components', '-n', default=20, type=int, help='number of component to output.')
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='size of each data batch.')
    parser.add_argument('--transform', '-t', default=0, type=int, choices=range(0, 2), help='transform data with PCA.')
    parser.add_argument('--log_flag', '-g', default=1, type=int, choices=range(0, 2), help="convert to log-scale")
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    logging.debug('Total data size: '+str(finfo.shape[0]))
    # Apply filter if specified
    if not args.filter is None:
        logging.debug('Read filter file: '+args.filter)
        flt = pd.read_csv(args.filter)
        logging.debug('Filter size: '+str(flt.shape[0]))
        fidx = finfo['timestamp'].isin(list(flt.iloc[:,0].apply(str)))
        finfo = finfo.loc[fidx,:].reset_index(drop=True)
        logging.debug('Data size after filter: '+str(finfo.shape[0]))
    # Check data dimension before proceed
    if finfo.shape[0] < args.n_components:
        logging.error('Number of data records is smaller than n_component, abort!')
        sys.exit('Number of data records is smaller than n_component, abort!')
    # Fit Incremental PCA
    logging.info("Performing IncrementalPCA of "+ str(args.n_components)+" components on data size of " + str(finfo.shape[0]) + " with batch size of " + str(args.batch_size) + "...")
    ipca = fit_ipca_partial(finfo, nc=args.n_components, bs=args.batch_size, log_flag=(args.log_flag==1))
    # Summarize results
    ev = ipca.explained_variance_
    evr = ipca.explained_variance_ratio_
    com = np.transpose(ipca.components_)
    logging.info("Explained variance ratio: "+ str(evr))
    # Output components
    #com_header = ['pc'+str(x+1) for x in range(args.n_components)]
    #writeToCsv(com, args.output+'.components.csv', header=com_header)
    pd.DataFrame({'pc':['pc'+str(i+1) for i in range(args.n_components)],'ev':ev, 'evr':evr}).to_csv(args.output+'.exp_var.csv', index=False)
    # Output fitted IPCA model
    with open(args.output+".pca.joblib", 'wb') as outfile:
        joblib.dump(ipca, outfile)
    # Transformed the training data if specified
    if args.transform==1:
        logging.info("Transform data with PCs...")
        # Transform data
        dbz_ipca = transform_dbz(ipca, finfo, to_log=True)
        # Append date and projections
        proj_header = ['timestamp'] + ['pc'+str(x+1) for x in range(args.n_components)]
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




