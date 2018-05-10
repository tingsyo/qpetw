#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TUtility script to preprocess RADAR data and group them into 1hr interval
"""
import os, csv, logging, argparse
import numpy as np
import pandas as pd

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2017~2018, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.0.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2018-04-20'

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
    
def read_dbz_memmap(finfo, tmpfile='dbz.dat', flush_cycle=14400):
    dbz = None
    fcount = 0
    # Setup numpy.memmap for storage
    tmp = read_dbz(finfo[0][0])
    logging.info("Start memmap with shape: ("+ str(len(finfo)) + "," + str(len(tmp)) +")")
    dbz = np.memmap(tmpfile, dtype='float32', mode='w+', shape=(len(finfo), len(tmp)))
    # Add 1st record
    dbz[0,:] = tmp[:]
    # Loop through finfo
    for i in range(1,len(finfo)):
        f = finfo[i]
        logging.debug('Reading data from: ' + f[0])
        tmp = read_dbz(f[0])
        # Append new record
        if tmp is None:     # Copy the previous record if new record is empty
            dbz[i,:] = dbz[(i-1),:]
        else:
            dbz[i,:] = tmp[:]
        # Flush memmap everry flush_cycle
        fcount+=1
        if fcount==flush_cycle:
            logging.debug("Flush memmap: "+str(fcount)+" | "+str(i))
            fcount = 0
            dbz.flush()
    # Save changes of the storage file
    dbz.flush()
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
    # Read into numpy.memmap
    logging.info("Extract data from all files: "+ str(len(finfo)))
    dbz = read_dbz_memmap(finfo)
    # Process dbz data with Incremental PCA
    logging.info("Performing IncrementalPCA with "+ str(args.n_components)+" components.")
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_size)
    dbz_ipca = ipca.fit_transform(dbz)
    evr = ipca.explained_variance_ratio_
    com = np.transpose(ipca.components_)
    logging.debug("Explained variance ratio: "+ str(evr))
    # Output components
    com_header = ['pc'+str(x+1) for x in range(args.n_components)]
    #np.insert(com, 0, com_header, axis=0)
    writeToCsv(com, args.output.replace('.csv','.components.csv'), header=com_header)
    # Append date and projections
    proj_header = ['date','hhmm'] + ['pc'+str(x+1) for x in range(args.n_components)]
    newrecs = []
    for i in range(len(finfo)):
        newrecs.append(finfo[i][1:3] + list(dbz_ipca[i]))
    # Output
    writeToCsv(newrecs, args.output, header=proj_header)
    # Save PCA for later use
    joblib.dump(ipca, args.output.replace(".csv",".pca.mod"))
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()




