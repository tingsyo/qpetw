#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads in the QPESUMS data in *.npy, parse its timstamp, and convert UTC to LST (UTC+8).
"""
import os, logging, argparse, datetime, struct
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
def search_cwbqpe_gz(srcdir, prefix='PCP_1H_RAD', ext='gz'):
    '''Scan specified directory for CWB pre-QC QPE data in *.gz format.
       In addition to list all URIs, this function also pasrses the filename for timestamp,
       and convert the timestamp from UTC to LST(UTC+8).
    '''
    import pandas as pd
    fileinfo = []
    for subdir, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.endswith('.gz'):                                   # Only work on 'gz' files
                furi = os.path.join(subdir, f)                      # Complete URI
                ftime = ''.join(f.split('.')[1:3])                  # time-stamp in YYYYmmddHHMM
                fileinfo.append({'furi':furi, 'timestamp':ftime})
    results = pd.DataFrame(fileinfo)
    results = results.sort_values(by=['timestamp']).reset_index(drop=True)
    results['lst'] = convert_utc_to_lst(results['timestamp'], time_zone=8)
    return(results)

def convert_utc_to_lst(tslist, time_zone=8):
    '''Convert a list of timestamps to the specified time zone (+8 by default).'''
    lst_list = []
    for t in tslist:
        t_utc = datetime.datetime.strptime(t,'%Y%m%d%H%M')      # Convert timestamp string yo datetime object
        t_lst = t_utc + datetime.timedelta(hours=time_zone)     # Change timezone
        lst_list.append(t_lst.strftime('%Y%m%d%H%M'))           # Convert datetime back to string
    return(lst_list)

def create_station_list(stinfo_uri, stid):
    '''Read in CWB station information and filter with selected IDs'''
    stlist = pd.read_csv(stinfo_uri)
    stlist = stlist.loc[stlist['id'].isin(stid),:].reset_index(drop=True)
    return(stlist)

def retrieve_cwbqpe(srcdir, stlist):
    '''Scan all CWB-QPE-pre-QC data, and retrieve precipitation of specified stations.'''
    # Scan source files
    srcinfo = search_cwbqpe_gz(srcdir)
    # Loop through source files
    results = []
    for i in range(srcinfo.shape[0]):
        timestamp = srcinfo['lst'].iloc[i]      # Retrieve timestamp in LST
        cq = cwbqpe(srcinfo['furi'].iloc[i])    # Load data
        try:
            cq.load_data()
        except struct.error:
            logging.error('Errors encountered while loading data from '+srcinfo['furi'].iloc[i])
            continue
        tmp ={'timestamp':timestamp}
        for i in range(stlist.shape[0]):    # Loop through stations
            sta = stlist.iloc[i,:]
            qval = cq.find_interpolated_value(sta['lon'], sta['lat'])
            tmp[sta['id']] = qval
        logging.debug(tmp['timestamp'])
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
    qpe_results = retrieve_cwbqpe(args.input, station_list)
    # Output results
    qpe_results.to_csv(args.output, index=False)
    # done
    return(0)

#==========
# Script
#==========
if __name__=="__main__":
    main()
