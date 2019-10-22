#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads a list of timestamps, and create a webpage to show figures that illustrate the QPESUMS data and CWB radar
"""
import os, csv, logging, argparse, time
from PIL import Image
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
def create_HTML_head():
	doc = '<!DOCTYPE html>\n<html>\n<body>\n'
	return(doc)

def create_HTML_tail():
    doc = '</body>\n</html>'
    return(doc)

def create_table(tslist, imgdir):
	# Create table head
	table_head = "<table>\n"
    # Loop through timestamp list
    table_body = ''
    for i in range(tslist.shape[0]):
    	# Parse timestamp and create uri
        t = tslist.timestamp.iloc[i].astype(str)
        uri_qpesums = imgdir+t+'_qpesums.jpg'
        uri_cwb = imgdir+t+'_cwb.jpg'
        # Create row
        row = "  <tr>\n"
        row += "    <td><img src=\'{0}\' /></td>\n".format(uri_qpesums)
        row += "    <td><img src=\'{0}\' /></td>\n  </tr>\n".format(uri_cwb)
        # Append row
        table_body += row
    # Complete table
    table = table_head + table_body + "</table>"
    return(table)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the filter file with time-stamps.')
    parser.add_argument('--imgdir', '-d', help='the directory containing downloaded images.')
    parser.add_argument('--output', '-o', default='output', help='the output directory.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Scan files for reading
    ts = pd.read_csv(args.input)
    # Create HTML
    html = create_HTML_head()
    html += create_table(ts, args.imgdir)
    html += create_HTML_tail()
    # Write html
    with open(args.output, 'w') as ofile:
    	ofile.write(html)
    # done
    return(0)

#==========
# Script
#==========
if __name__=="__main__":
    main()

