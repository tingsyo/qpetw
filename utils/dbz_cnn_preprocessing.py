#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script read RADAR-dbz data in text format and convert to a dictionary of length T 
where each element represents a 6*275*162 numpy array
"""
import os, csv, logging, argparse, pickle
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
def getDictHHMM():
    ''' Given a timestamp string in the form of YYYYMMDDHH00, return the 6 hhmm strings prior to the specified time. '''
    hhlist = ["00","01","02","03","04","05","06","07","08","09","10","11",
                  "12","13","14","15","16","17","18","19","20","21","22","23"]
    mmlist = ["00","10","20","30","40","50"]
    hhmm = []
    for h in hhlist:
        for m in mmlist:
            hhmm.append(h+m)
    # Target hour
    thh = ["01","02","03","04","05","06","07","08","09","10","11","12",
           "13","14","15","16","17","18","19","20","21","22","23","24"]
    # Create dictionary
    hhmmdict = {}
    for i in range(24):
        hhmmdict[thh[i]] = hhmm[i*6:i*6+6]
    # Done
    return(hhmmdict)


def search_dbz(srcdir):
    import os
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
    import pandas as pd
    import numpy as np
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2])).reshape((275,162))
    except pd.errors.EmptyDataError:
        logging.warning(furi + " is empty.")
    return(results)

def create_input_from_dir(sdir):
    ''' Create a stack of input data from given directory containing dbz data '''
    finfo = search_dbz(sdir)            # Scan and parse file names
    fdays = [d[1] for d in finfo]       # Retrieve dates
    days = sorted(list(set(fdays)))     # Clean up dates
    # HHMM dictionary
    hhmm = getDictHHMM()
    hh = list(hhmm.keys())
    # Create input label: YYYYMMDDHH
    fdf = pd.DataFrame(finfo, columns=['furi','day','hhmm'])
    ilab = []
    flist = []
    x = []
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
                print(d+h+" data is complete, reading data...")
                flist.append(f6)
                ilab.append(d+h)
                for i in range(6):
                    dbz = read_dbz(f6[i])
                    xtmp.append(dbz)
                # Append 6*275*162 array to the list
                #print(np.float32(np.array(xtmp)).shape)
                x.append(np.float32(np.array(xtmp)))
            else:
                print("Time-flag "+d+h+" contains missing data, number of records: "+str(len(f6)))
    #
    return(ilab, flist, x)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing all the DBZ data.')
    parser.add_argument('--output', '-o', default='output.npy', help='the output file.')
    parser.add_argument('--log', '-l', default='tmp.log', help='the log file.')
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG)
    # Create data list
    ts, flist, data = create_input_from_dir(args.input)
    # Write out
    np.save(args.output, data)
    with open(args.output+'.csv', 'w') as cf:
        cw = csv.writer(cf, delimiter=',')
        cw.writerow(['time','f1','f2','f3','f4','f5','f6'])
        for i in range(len(ts)):
            rec = [ts[i]]+flist[i]
            cw.writerow(rec)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
