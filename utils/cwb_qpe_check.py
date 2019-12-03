#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads in the QPESUMS data in *.npy, parse its timstamp, and convert UTC to LST (UTC+8).
"""
import os, logging, argparse, datetime, shutil
import numpy as np
import pandas as pd
from cwbqpe import *
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
# Parameters: 45 station ids in TPE
STID_TPE45 = ['466880', '466910', '466920', '466930', '466940', 
              'C0A520', 'C0A530', 'C0A540', 'C0A550', 'C0A560', 
              'C0A570', 'C0A580', 'C0A640', 'C0A650', 'C0A660', 
              'C0A710', 'C0A860', 'C0A870', 'C0A880', 'C0A890', 
              'C0A920', 'C0A940', 'C0A950', 'C0A970', 'C0A980', 
              'C0A9A0', 'C0A9B0', 'C0A9C0', 'C0A9E0', 'C0A9F0', 
              'C0A9G0', 'C0A9I1', 'C0AC40', 'C0AC60', 'C0AC70', 
              'C0AC80', 'C0ACA0', 'C0AD00', 'C0AD10', 'C0AD20', 
              'C0AD30', 'C0AD40', 'C0AD50', 'C0AG90', 'C0AH00']
#
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

def create_station_list(stinfo_uri, stid):
    stlist = pd.read_csv(stinfo_uri)
    stlist = stlist.loc[stlist['id'].isin(stid),:].reset_index(drop=True)
    return(stlist)

def parse_qpe_filename(furi):
    ts = ''.join(furi.split('.')[1:3])
    return(ts)

def retrieve_qpe(srcdir, stlist):
    # Scan source files

    # Loop through source files
    results = []
    for f in flist:
        timestamp = parse_qpe_filename(f)   # Parse file name for time-stamp
        cq = cwbqpe(f)                      # Load data
        cq.load_data()
        tmp = {}
        for i in range(stlist.shape[0]):    # Loop through stations
            sta = stlist.iloc[i,:]
            qval = cq.find_interpolated_value(sta['lon'], sta['lat'])
            tmp[sta['id']] = qval
        results.append(tmp)
    #
    return(pd.DataFrame(results))


#-----------------------------------------------------------------------
def main():
    '''
  1. Create station list
    - Scan *.csv files in precipitation.tpe for IDs
    - Read station list (lon/lat/lev...)
    - Intersect IDs and station_list
  2. Scan cwbqpe data
    - Scan for all available CWB_QPE data -> cqlist
    - Loop through cqlist
      - load binary
      - Convert time-stamp to LST
      - get values of each station
      - Save as table: timestamp-by-station_id
  3. Validation
    - Verify the derived QPE results with precipitation.tpe/*.csv
    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing all CWB_pre_QC_QPE data.')
    parser.add_argument('--output', '-o', default='output', help='the output file.')
    parser.add_argument('--station_list', '-s', help='the file containing all weather station information.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # 1. Create station list
    station_list = create_station_list(args.station_list, STID_TPE45)
    # 2. Scan cwbqpe data
    qpe_results = retrieve_qpe(args.input, station_list)
    # Output results
    qpe_results.to_csv(args.output, index=False)
    # done
    return(0)

#==========
# Script
#==========
if __name__=="__main__":
    main()
