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
import os, csv, logging, argparse, pickle, h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

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
        xdata.append(tmp)
    x = np.array(xdata, dtype=np.float32)
    return(x)
    

''' Time-stamp processing '''
# Convert HH=24 to HH=00
def correct_qpe_timestamp(ts):
    '''Check the time-stamp string in the form of YYYY-mm-dd-HH:
         - if HH = 24, increment the dd by one and change HH to 00
    '''
    import datetime
    if ts[8:] == '24':
        oldt = datetime.datetime.strptime(ts[:8], '%Y%m%d')
        newt = oldt + datetime.timedelta(days=1)
        newt_str = newt.strftime('%Y%m%d')+'00'
        return(newt_str)
    else:
        return(ts)
    
# Convert HH=00 to HH=24
def convert_to_qpe_timestamp(ts):
    '''Check the time-stamp string in the form of YYYY-mm-dd-HH:
         - if HH = 00, decrease the dd by one and change HH to 24
    '''
    import datetime
    if ts[8:] == '00':
        oldt = datetime.datetime.strptime(ts[:8], '%Y%m%d')
        newt = oldt - datetime.timedelta(days=1)
        newt_str = newt.strftime('%Y%m%d')+'24'
        return(newt_str)
    else:
        return(ts)

 # Create a list of YYYYMMDDHH with 1hour interval during the start_date and end_date.
def create_timestamps(start_date, end_date, nhr=1):
    '''Creating a list of YYYYMMDDHH with 1hour interval during the start_date and end_date.'''
    import datetime
    # Convert YYYYMMDD strings to datetime object
    starttime = datetime.datetime.strptime(start_date, '%Y%m%d%H')
    endtime = datetime.datetime.strptime(end_date, '%Y%m%d%H')
    timestep = datetime.timedelta(hours=nhr)
    # Create YYYYMMDDHH list 
    tslist = []
    while starttime <= endtime:
        tslist.append(convert_to_qpe_timestamp(starttime.strftime('%Y%m%d%H')))
        starttime += timestep
    # Done
    return(tslist)

''' Perform Incremental PCA '''
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
            f = finfo.iloc[j,:]
            logging.debug('Reading data from: ' + f['furi'])
            tmp = np.load(f['furi'])
            # Append new record
            if tmp is not None:            # Append the flattened data array if it is not None
                dbz.append(tmp.flatten())
        # Partial fit with batch data
        dbz = np.array(dbz)
        print(dbz.shape)                   # debug information
        ipca.partial_fit(dbz)
    # done
    return(ipca)

''' Project data into PCs '''
def transform_dbz(ipca, finfo):
    dbz = []
    # Loop through finfo
    for i in range(0,len(finfo)):
        f = finfo.iloc[i,:]
        logging.debug('Reading data from: ' + f['furi'])
        tmp = np.load(f['furi']).flatten()
        # Append new record
        if tmp is None:     # Copy the previous record if new record is empty
            print('File empty: '+f['furi'])
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
    parser.add_argument('--filter', '-f', default=None, help='the filter file with time-stamps.')
    parser.add_argument('--n_components', '-n', default=20, type=int, help='number of component to output.')
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='size of each data batch.')
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    print('Total data size: '+str(finfo.shape[0]))
    # Apply filter if specified
    if not args.filter is None:
        print('Read filter file: '+args.filter)
        flt = pd.read_csv(args.filter)
        print('Filter size: '+str(flt.shape[0]))
        fidx = finfo['timestamp'].isin(list(flt.iloc[:,0].apply(str)))
        finfo = finfo.loc[fidx,:].reset_index(drop=True)
        print('Data size after filter: '+str(finfo.shape[0]))
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
    with open(args.output.replace(".csv",".pca.mod"), 'wb') as outfile:
        pickle.dump(ipca, outfile)

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




