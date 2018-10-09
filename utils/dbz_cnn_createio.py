#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script read preprocessed RADAR-dbz data and precipitation data,
and then create paired input-output dataset.
"""
import os, csv, logging, argparse, h5py, glob
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
__date__ = '2018-10-04'

# Parameters
nSample = 37369                 # Total complete samples
nLayer = 6                      # 6 10-min dbz for an hour
nY = 275                        # y-dimension of dbz data
nX = 162                        # x-dimension of dbz data
batchSize = 128                 # Batch size for training / testing
yseg_stat = [0.5, 8, 13, 29]    # 40-year statistics of hourly precipitation of trace, 90%, 95%, and 99% percentile
yseg = [0.5, 10, 15, 30]        # Actual segmentation for precipitation

# Functions
def matchDate(xdate, y, hourTag='t1hr', timeLag=False):
    ''' Given the input/output data, check for data completeness and match records by specified conditions '''
    results = None
    ydate = np.array(y['date'], dtype='str')
    vd = [x for x in xdate if x in ydate]
    return(results)

def create_input_by_hour(srcdir, outdir):
    ''' Create a stack of input data from given directory containing dbz data '''
    finfo = search_dbz(srcdir)            # Scan and parse file names
    fdays = [d[1] for d in finfo]       # Retrieve dates
    days = sorted(list(set(fdays)))     # Clean up dates
    # HHMM dictionary
    hhmm = getDictHHMM()
    hh = list(hhmm.keys())
    # Create input label: YYYYMMDDHH
    fdf = pd.DataFrame(finfo, columns=['furi','day','hhmm'])
    ilab = []
    flist = []
    # Loop through days
    for d in days:
        tmp = [f for f in finfo if f[1]==d]
        # Loop through hours
        for h in hh:
            f6 = [t[0] for t in tmp if t[2] in hhmm[h]]
            f6 = sorted(f6)
            # Read data if the 6-file-set is complete
            xtmp = []
            if(len(f6)==6):
                print("Time-flag "+d+h+" data is complete, reading data...")
                flist.append(f6)
                ilab.append(d+h)
                for i in range(6):
                    dbz = read_dbz(f6[i])
                    xtmp.append(dbz)
                # Append 6*275*162 array to the list
                np.save(outdir+"/"+d+h+".npy", np.float32(np.array(xtmp)))
            else:
                print("Time-flag "+d+h+" contains missing data, number of records: "+str(len(f6)))
    #
    return(ilab, flist)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing preprocessed DBZ data.')
    parser.add_argument('--rawy', '-y', help='the file containing the precipitation data.')
    parser.add_argument('--input', '-i', default='x.hdf5', help='the processed input data.')
    parser.add_argument('--output', '-o', default='y.hdf5', help='the processed output data.')
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Read raw input and output
    logging.info("Reading input X from: "+ args.inputx)
    xfiles = glob.glob(args.rawx+'/*.npy')
    logging.info("Reading output Y from: "+ args.inputy)
    yraw = pd.read_csv(args.rawy)
    # Find intersect of input/output dates
    xdate = [f.split('.')[0].split('/')[1] for f in xfiles]
    ydate = np.array(yraw['date'], dtype='str')
    valid_date = matchDate()   
    nSample = len(valid_date)
    # Create paired Input data
    fx = h5py.File(args, 'w')
    xdata = fx.create_dataset('x', (nSample, nLayer, nY, nX), chunks=(batchSize, nLayer, nY, nX), dtype=np.float32)
    for i in range(nSample):
        tmp = np.load(args.raws+'/'+valid_date[i]+'.npy')
        xdata[i] = tmp
    # Create paired Output data
    
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
