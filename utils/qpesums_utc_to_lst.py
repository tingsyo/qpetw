#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads in the QPESUMS data in *.npy, parse its timstamp, and convert UTC to LST (UTC+8).
"""
import os, logging, argparse, datetime, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------
__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2017~2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-01-09'
#-----------------------------------------------------------------------
def search_qpesums_npy(srcdir):
    '''Scan QPESUMS data in *.npy format (6*275*162) from the specified directory.
    '''
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

def correct_qpesums_datetime(ts):
    '''Check the time-stamp string in the form of YYYY-mm-dd-HH:
         - if HH = 24, increment the dd by one and change HH to 00
    '''
    import datetime
    if ts[8:] == '24':
        oldt = datetime.datetime.strptime(ts[:8], '%Y%m%d')
        newt = oldt + datetime.timedelta(days=1)
    else:
        newt = datetime.datetime.strptime(ts, '%Y%m%d%H')
    return(newt)

def covert_utc_to_lst(tslist, time_zone=8):
    '''Convert a list of timestamps to the specified time zone (+8 by default).
    '''
    lst_list = []
    for t in tslist:
        t_utc = correct_qpesums_datetime(t)                     # Correct hour naming
        t_lst = t_utc + datetime.timedelta(hours=time_zone)     # Shift time zone
        lst_list.append(t_lst.strftime('%Y%m%d%H'))             # Convert to string
    return(lst_list)

def correct_qpesums_files(finfo, outdir):
    '''Copy the original QPESUMS data to the new directory with the corresponding timestamp in LST.
    '''
    for i in range(finfo.shape[0]):
        rec = finfo.iloc[i,:]
        newuri = os.path.join(outdir, rec['lst']+'.npy')
        logging.debug("Copying " + rec['furi'] + ' to ' + newuri)
        shutil.copy(rec['furi'], newuri)
    return(0)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing all the DBZ data.')
    parser.add_argument('--output', '-o', default='output', help='the output directory.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Scan files for reading
    finfo = search_qpesums_npy(args.input)
    # Create LST timestamp
    finfo['lst'] = covert_utc_to_lst(finfo.timestamp)
    # Generate output
    correct_qpesums_files(finfo, args.output)
    # done
    return(0)

#==========
# Script
#==========
if __name__=="__main__":
    main()
