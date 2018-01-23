#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script of using Keras to implement a 1D convolutional neural network (CNN) for regression.
"""
import os, csv, logging, argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

def read_dbz(srcdir):
    fileinfo = []
    results = []
    for subdir, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.endswith('.txt'):
                logging.debug(f)
                # Parse file name for time information
                finfo = f.split('.')
                fileinfo.append(finfo[1:3])
                # Read data through pandas.read_fwf
                tmp = pd.read_fwf(os.path.join(subdir, f), widths=[8,8,8], header=None)
                results.append(tmp.iloc[:,2])
    return((fileinfo, np.array(results)))


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
    # Read iodata
    finfo, dbz = read_dbz(args.input)
    # Process dbz data with Incremental PCA
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=200)
    dbz_ipca = ipca.fit_transform(dbz)
    evr = ipca.explained_variance_ratio_
    com = ipca.components_
    # Output components and projections
    output = []
    for i in range(len(evr)):
        output.append([evr[i]]+com[i])
    writeToCsv(output, args.output.replace('.csv','.components.csv'))
    # Append date and projections
    newrecs = []
    for i in range(len(recs['date'])):
        newrecs.append([recs['date'][i]] + list(dbz_ipca[i]))
    # Output
    writeToCsv(newrecs, args.output)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()




