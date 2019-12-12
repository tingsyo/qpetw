#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script to preprocess RADAR data:
- Read in QPESUMS data in text format (ny=275, nx=162).
- Group 6 10-min files into one numpy array of 1hr interval (y-x-t).
"""
import os, csv, logging, argparse
import numpy as np
import pandas as pd

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-10-27'

def read_qpesums_text(furi):
    '''Read the QPESUMS data provided by NCDR. It is stored as plain text in fixed-width-format.
       Three data columns are longitude, latitude, and the maximal reflectivity.
    '''
    import pandas as pd
    import numpy as np
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2])).reshape((275,162))
    except pd.errors.EmptyDataError:
        logging.error(furi + " is empty.")
    return(results)

def search_qpesums_text(srcdir, prefix='COMPREF.', ext='.txt'):
    '''Search the specified the directory for QPESUMS data in text format and retrieve related information.
         example file name: COMPREF.20160101.0010.txt
    '''
    import os
    import pandas as pd
    fileinfo = []
    # Walk through the specified directory
    for root, dirs, files in os.walk(srcdir, followlinks=True):
        for f in files:
            if f.startswith(prefix) and f.endswith(ext):
                # Parse file name for time information
                furi = os.path.join(root, f)
                timestr = f.replace(prefix,'').replace(ext,'')
                # Append data
                fileinfo.append({'furi':furi, 'timestamp':timestr})
    # Return the results as a DataFrame
    results = pd.DataFrame(fileinfo).sort_values('timestamp').reset_index(drop=True)
    return(results)

def parse_time_string(tstr, fmt='%Y%m%d.%H%M'):
    '''Parse a time string into a datetime object. The given format was YYYYMMDD.HHMM.
       This function is separated in case the format of time-string might change over time.
    '''
    import datetime
    tobj = datetime.datetime.strptime(tstr, fmt)
    return(tobj)

def aggregate_qpesums_data(current_time, data_info, time_shift=0):
    ''' - check the availability of involved data
        - perform aggregation (y, x, t) if available
        - time-zone conversion if necessary
    '''
    from datetime import timedelta, date
    import numpy as np
    import pandas as pd
    # Get the previous 6 time-stamps with 10-min interval
    rec = {'timestamp': current_time.strftime("%Y%m%d.%H")}
    fcount = 0
    tslist = []
    for i in range(6,0,-1):
        ts = (current_time-timedelta(minutes=(i*10))).strftime("%Y%m%d.%H%M")
        # Check availability of required time-stamps
        rec['m-'+str((i-1)*10)] = (ts in data_info.timestamp.values)*1
        fcount += rec['m-'+str((i-1)*10)]
        tslist.append(ts)
    # Check data completeness
    rec['complete'] = (fcount==6)
    qdata = None
    if fcount==6:
        logging.debug('Aggregate data for '+current_time.strftime("%Y%m%d.%H"))
        qdata = []
        for ts in tslist:
            logging.debug('....reading data for '+ts)
            tmp = read_qpesums_text(data_info.loc[data_info.timestamp==ts, 'furi'].iloc[0])
            # Break if the given hour has missing data
            if tmp is None:
                logging.warning('Empty data found in '+current_time.strftime("%Y%m%d.%H")+', break.')
                qdata = None
                break
            qdata.append(tmp)
        qdata =  np.stack(qdata, axis=2)
    else:
        logging.error('Data missing for '+current_time.strftime("%Y%m%d.%H"))
    # Shift time if necessary
    if time_shift !=0:
        rec['timestamp'] = (current_time+timedelta(hours=time_shift)).strftime("%Y%m%d.%H")
    return((rec, qdata))

def merge_10min_to_hourly(data_info, outdir, time_shift=0):
    '''The main function of merging 6 10-min data files into one hourly data array.
    '''
    import numpy as np
    from datetime import timedelta, datetime
    # Set up the start-time and end-time
    starttime = parse_time_string(data_info.timestamp.iloc[0])
    endtime = parse_time_string(data_info.timestamp.iloc[-1])
    logging.info('Aggregate QPESUMS data from '+starttime.strftime("%Y%m%d.%H%M")+' to '+endtime.strftime("%Y%m%d.%H%M"))
    # Loop through time-stamps by hour
    dclist = []
    ctime = starttime+timedelta(hours=1)
    ctime = ctime.replace(minute=0, second=0, microsecond=0)
    while ctime <= (endtime+timedelta(hours=1)):
        check, data = aggregate_qpesums_data(ctime, data_info, time_shift=time_shift)
        dclist.append(check)
        # Output the aggregated data to the outdir
        if not data is None:
            ofname = outdir + '/' + check['timestamp'].replace('.','') + '.npy'
            np.save(ofname, data)
        else:
            logging.error('Data is empty at '+check['timestamp'])
        # Move forward by one hour
        ctime += timedelta(hours=1)
    #
    return(pd.DataFrame(dclist))


#-----------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve 10-min QPESUMS data in TEXT format and convert into 1-hour y-x-t numpy arrays.')
    parser.add_argument('--input', '-i', help='the directory containing the QPESUMS data.')
    parser.add_argument('--output', '-o', help='the directory to store the output files.')
    parser.add_argument('--log', '-l', default=None, help='the log file.')
    parser.add_argument('--prefix', '-p', default='COMPREF.', help='the prefix of the QPESUMS data files.')
    parser.add_argument('--ext', '-e', default='.txt', help='the extension of the QPESUMS data files.')
    parser.add_argument('--timeshift', '-s', default=8, help='Adjustment of the time zone. 8 by default, means change from UTC to LST of Taipei.')
    args = parser.parse_args()
    # Set up logging
    if not args.log is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.log, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Scan and parse qpesums data files
    logging.info('Searching QPESUMS data files in ' + args.input)
    finfo = search_qpesums_text(args.input, prefix=args.prefix, ext=args.ext)
    logging.info('Totally data files found: ' + str(finfo.shape[0]))
    # Data processing and output
    merge_10min_to_hourly(finfo, args.output, args.timeshift)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
