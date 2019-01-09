#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads RADAR-dbz data in text format (275*162 values with lon/lat), and performs 
convolutional autoencoder algorithm to reduce the data diemnsion. 
"""
import os, csv, logging, argparse, pickle
import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2017~2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "UNLICENSED"
__version__ = "0.0.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-01-09'

# Utilities
def search_dbz(srcdir):
    '''
    Search the specified directory and list all dbz file ends with '.txt'.
    '''
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
    '''
    Read in dbz data from specified fix-width plain text file.
    '''
    import pandas as pd
    import numpy as np
    results = None
    try:
        tmp = pd.read_fwf(furi, widths=[8,8,8], header=None)
        results = np.float32(np.array(tmp.iloc[:,2])).reshape((275,162))
    except pd.errors.EmptyDataError:
        logging.warning(furi + " is empty.")
    return(results)

#

