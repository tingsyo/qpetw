#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script of using Keras to implement a 1D convolutional neural network (CNN) for regression.
"""
import os, csv, logging, argparse
from tempfile import mkdtemp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

def search_dbz(srcdir):
    fileinfo = []
    results = []
    for subdir, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.endswith('.txt'):
                # Parse file name for time information
                furi = os.path.join(subdir, f)
                finfo = f.split('.')
                logging.debug([furi] + finfo[1:3])
                fileinfo.append([furi] + finfo[1:3])
    return(fileinfo)

def read_dbz(furi):
    tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
    results = np.float32(np.array(tmp.iloc[:,2]))
    return(results)
    
def read_dbz_memmap(finfo):
    tmpfile = 'dbz.dat'
    dbz = None
    for f in finfo:
        logging.debug('Reading data from: ' + f[0])
        tmp = read_dbz(f[0])
        if dbz is None:
            logging.debug("Start memmap with shape: "+ str(tmp.shape))
            dbz = np.memmap(tmpfile, dtype='float32', mode='w+', shape=tmp.shape)
            dbz = tmp
        else:
            dbz = np.vstack((dbz, tmp))
            logging.debug("Appending numpy.memmap: "+ str(dbz.shape))
    return(dbz)

def writeToCsv(output, fname):
    # Overwrite the output file:
    with open(fname, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')#,quotechar='"', quoting=csv.QUOTE_ALL)
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
    parser.add_argument('--n_components', '-n', default=50, type=int, help='number of component to output.')
    parser.add_argument("-r", "--randomseed", help="integer as the random seed", default="12321")
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Scan files for reading
    finfo = search_dbz(args.input)
    # Read into numpy.memmap
    dbz = read_dbz_memmap(finfo)
    # Process dbz data with Incremental PCA
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=100)
    dbz_ipca = ipca.fit_transform(dbz)
    evr = ipca.explained_variance_ratio_
    com = ipca.components_
    print("Explained variance ratio: "+ str(evr))
    # Output components and projections
    output = []
    for i in range(len(evr)):
        output.append([evr[i]]+com[i])
    writeToCsv(output, args.output.replace('.csv','.components.csv'))
    # Append date and projections
    newrecs = []
    for i in range(len(finfo)):
        newrecs.append(finfo[i][1:3] + list(dbz_ipca[i]))
    # Output
    writeToCsv(newrecs, args.output)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()




