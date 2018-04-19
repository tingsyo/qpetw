#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: run.qpetpe.py
Version: 1.0.0
Date: 2018-04-19
Description:
  Perform QPE for 20 weather stations in Northern Taiwan
"""
import os, csv, logging, argparse, time,sys
import numpy as np
# Parameters
station_id = ["466880", "466900", "466910", "466920", "466930", "466940", "466950", "C0A520", "C0A530", "C0A540",
 "C0A550", "C0A560", "C0A570", "C0A580", "C0A640", "C0A650", "C0A660", "C0A710", "C0A860", "C0A870",
 "C0A880", "C0A890", "C0A920", "C0A931", "C0A940", "C0A950", "C0A970", "C0A980", "C0A9A0", "C0A9B0",
 "C0A9C0", "C0A9E0", "C0A9F0", "C0A9G0", "C0A9I1", "C0AC40", "C0AC60", "C0AC70", "C0AC80", "C0ACA0",
 "C0AD00", "C0AD10", "C0AD20", "C0AD30", "C0AD40", "C0AD50", "C0AG90", "C0AH00", "C0AH10"]

# Setup progressbar
def setup_progressbar(pb_width):
    sleeptime = np.abs(np.random.randn(pb_width))
    sleeptime = sleeptime/np.max(sleeptime)/2
    # Header
    sys.stdout.write("[%s]" % (" " * pb_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (pb_width+1)) # return to start of line, after '['
    # Loop through width
    for i in range(pb_width):
        time.sleep(sleeptime[i])
        sys.stdout.write("-")
        sys.stdout.flush()
    # End
    sys.stdout.write("\n")
    return(0)
    
def performQPE(id, upperLimit=50, lowerLimit=0):
    # Header
    print('Quantitative Precipitation Estimation for station: '+id)
    # Set up progress
    setup_progressbar(40)
    # Simulate results
    pred = np.random.gamma(shape=1.0, scale=5.0)
    print('Predicted Precipitation: ' + str(pred))
    return(0)

#-----------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Predict precipitation for specified stations.')
    parser.add_argument('--input', '-i', default=None, help='the input data.')
    parser.add_argument('--output', '-o', default='output.csv', help='the output file.')
    parser.add_argument('--n_station', '-n', default=20, type=int, help='number of stations to output.')
    parser.add_argument('--log', '-l', default='qpetpe.log', help='the log file.')
    args = parser.parse_args()
    # Check arguments
    if args.n_station > len(station_id):
        args.n_station = len(station_id)
    # Process input
    if args.input is not None:
        print("processing input file: " + args.input)
        setup_progressbar(60)
    # Loop through stations:
    for i in range(args.n_station):
        performQPE(station_id[i])
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()


