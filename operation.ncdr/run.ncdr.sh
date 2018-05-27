#!/bin/bash
# Operation script for calling from schedule
source qpetw.cfg
dt=$(date '+%Y%m%d%H00')
#-----------------------------------------------------------------------
# Step 1: Data aggregation
#-----------------------------------------------------------------------
#python3 lib/qpetw_check.py
#-----------------------------------------------------------------------
# Step 2: Data preprocessing
#-----------------------------------------------------------------------
python3 lib/qpetw_preprocessing.py -i $DBZ_RAW_PATH -o $DBZ_INPUT -m $MOD_DBZPCA

#-----------------------------------------------------------------------
# Step 3: Prediction
#-----------------------------------------------------------------------
Rscript --vanilla lib/qpetw_predict.r

