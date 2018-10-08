#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script read preprocessed RADAR-dbz data and precipitation data,
and then create paired input-output dataset.
"""
import os, csv, logging, argparse, h5py
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
def get_dbz_info(dbzdir):
    import os
    fileinfo = []
    results = []
    for subdir, dirs, files in os.walk(dbzdir, followlinks=True):
        for f in files:
            if f.endswith('.npy'):
                # Parse file name for time information
                furi = os.path.join(subdir, f)
                finfo = f.split('.')
                #logging.debug([furi] + finfo[1:3])
                fileinfo.append([furi] + finfo[1:3])
    return(fileinfo)


def read_precipitation(furi):
    import pandas as pd
    import numpy as np
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2])).reshape((275,162))
    except pd.errors.EmptyDataError:
        logging.warning(furi + " is empty.")
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
    parser.add_argument('--input', '-i', default='x.h5py', help='the processed input data.')
    parser.add_argument('--output', '-o', default='y.h5py', help='the processed output data.')
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Read raw input and output
    logging.info("Reading input X from: "+ args.inputx)
    xinfo = pd.read_csv(args.rawx)
    logging.info("Reading output Y from: "+ args.inputy)
    yraw = pd.read_csv(args.rawy)
    # Find intersect of input/output dates
    xdate = np.array(xinfo['time'], dtype='str')
    ydate = np.array(yraw['date'], dtype='str')
    valid_date = [x for x in xdate if x in ydate]
    # Create data list
    
    # Write out
    
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
