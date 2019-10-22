#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads a list of time stamps, and create figures that illustrate the QPESUMS data
"""
import os, csv, logging, argparse, time
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
def correct_qpe_timestamp(ts):
    '''Check the time-stamp string in the form of YYYY-mm-dd-HH:
         - if HH = 24, increment the dd by one and change HH to 00
    '''
    import datetime
    if ts[8:] == '24':
        oldt = datetime.datetime.strptime(ts[:8], '%Y%m%d')
        newt = oldt + datetime.timedelta(days=1)
        newt_str = newt.strftime('%Y%m%d')+'00'
        return(newt_str)
    else:
        return(ts)
    
def convert_to_qpe_timestamp(ts):
    '''Check the time-stamp string in the form of YYYY-mm-dd-HH:
         - if HH = 00, decrease the dd by one and change HH to 24
    '''
    import datetime
    if ts[8:] == '00':
        oldt = datetime.datetime.strptime(ts[:8], '%Y%m%d')
        newt = oldt - datetime.timedelta(days=1)
        newt_str = newt.strftime('%Y%m%d')+'24'
        return(newt_str)
    else:
        return(ts) 

def plot_qpesums(data, outfile=None):
    # Import library
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    # Define map boundary
    lat0 = 21.8875
    lat1 = 25.3125
    lon0 = 120.0
    lon1 = 122.0125
    lats = np.arange(21.8875, 25.3125, 0.0125)
    lons = np.arange(120.0, 122.0125, 0.0125)
    # Get data dimensions
    nl, ny, nx = data.shape
    print('Data dimensions: ' + str(nl) + ' layers of ' + str(ny) + ' x ' + str(nx))
    # Making plot
    m = Basemap(llcrnrlon=lon0, urcrnrlon=lon1, llcrnrlat=lat0, urcrnrlat=lat1, resolution='l')     # create basemap
    m.drawcoastlines()                              # draw coastlines on map.
    m.imshow(data[nl-1], alpha=0.99, cmap='Greys')  # fill data
    m.colorbar()                                    # add colobar
    # Turn off axis labels
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # Wrapping up
    plt.tight_layout()
    # Write output if specified
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
    # Done
    return(0)

def get_nccudb_cwbrad(timestamp, outfile=None):
    ''' Fetch the CWB radar iamge stored in NCCU SSL database. 
      - CCU/SSL URL naming examples:
          - http://140.137.32.24/aweb/cwbrad/2013/06/15/20130615_1500.cwbrad.2MOSSSL.jpg
          - http://140.137.32.24/aweb/cwbrad/2014/06/26/20140626_1500.cwbrad.2MOSSSL.jpg
          - http://140.137.32.24/aweb/cwbrad/2015/08/07/20150807_0400.cwbrad.2MOSSSL.jpg
          - http://140.137.32.24/aweb/cwbrad/2016/04/18/20160418_0500.cwbrad.2MOSSSL.jpg
    '''
    import urllib
    # DB parameters
    head_url = 'http://140.137.32.24/aweb/cwbrad/'
    tail_url = '00.cwbrad.2MOSSSL.jpg'
    # Fix timestamp if necessary
    timestamp = correct_qpe_timestamp(timestamp)
    # Parse the given time stamp
    yyyy = timestamp[0:4]
    mm = timestamp[4:6]
    dd = timestamp[6:8]
    hh = timestamp[8:10]
    # Make full url
    full_url = head_url + yyyy + '/' + mm + '/' + dd + '/' + yyyy+mm+dd+'_'+hh + tail_url
    # Get file
    if outfile is None:     # Return jpg file as bytes
        data = urllib.request.urlopen(full_url).read()
    else:                   # Save file and return results
        data = urllib.request.urlretrieve(full_url, outfile)
    return((full_url, data))

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing all the DBZ data.')
    parser.add_argument('--filter', '-f', help='the filter file with time-stamps.')
    parser.add_argument('--output', '-o', default='output', help='the output directory.')
    parser.add_argument('--randomseed', '-r', help="integer as the random seed", default="1234543")
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Scan files for reading
    ts = pd.read_csv(args.filter)
    # Loop through timestamp list
    for i in range(ts.shape[0]):
        t = ts.timestamp.iloc[i].astype(str)
        # Load qpesums data
        logging.info('Read QPESUMS data from '+ args.input + t + '.npy')
        qpsdata = np.load(args.input+t+'.npy')
        plot_qpesums(qpsdata, outfile=args.output+'/'+t+'_qpesums.jpg')
        # Download 
        logging.info('Read CWB Radar image from NCCU database')
        get_nccudb_cwbrad(t, outfile=args.output+'/'+t+'_cwb.jpg')
        # Sleep
        time.sleep(3)
    # done
    return(0)

#==========
# Script
#==========
if __name__=="__main__":
    main()
