#!/usr/bin/bash
# Operation script for calling from schedule
source(qpetw.cfg)
#-----------------------------------------------------------------------
# Step 1: Data aggregation
#-----------------------------------------------------------------------
python3 lib/qpetw_check.py
#-----------------------------------------------------------------------
# Step 2: Data preprocessing
#-----------------------------------------------------------------------
python3 lib/qpetw_preprocessing.py -i $DATASRC -o $QPETW_INPUT_FILE -m $QPETW_PCA_FILE

#-----------------------------------------------------------------------
# Step 3: Prediction
#-----------------------------------------------------------------------
Rscript --vanilla lib/qpetw_predict.r

