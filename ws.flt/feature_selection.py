#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose: Perform Incremental PCA on QPESUMS data.
"""
import os, csv, logging, argparse, pickle, h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
import joblib

# Read precipitation data
tmp = pd.read_csv('../examples/data/t1hr.csv')
# Calculate maximal precipitation among 45 stations
dates = tmp['date']
t1hr = tmp.iloc[:,1:]
t1hr_max = pd.DataFrame({'timestamp':dates, 'prec':t1hr.max(axis=1)})

# Read projections
pfull = pd.read_csv('../ws.flt/log/proj_ln_full.csv')
pfp40 = pd.read_csv('../ws.flt/log/proj_ln_fp40.csv')
pftyw = pd.read_csv('../ws.flt/log/proj_ln_ftyw.csv')

# Create new column names
pfull.columns = ['full_'+i for i in pfull.columns]
pfp40.columns = ['fp40_'+i for i in pfp40.columns]
pftyw.columns = ['ftyw_'+i for i in pftyw.columns]
pfull.rename(columns={'full_timestamp':'timestamp'}, inplace=True)
pfp40.rename(columns={'fp40_timestamp':'timestamp'}, inplace=True)
pftyw.rename(columns={'ftyw_timestamp':'timestamp'}, inplace=True)

# Clean up
data = pd.merge(pd.merge(pfull.iloc[:,:11], pfp40.iloc[:,:11], on='timestamp'),pftyw.iloc[:,:11], on='timestamp')
data.index = list(data['timestamp'])
data = data.iloc[:,1:]

# Add logarithm
for i in range(data.shape[1]):
    data['squared_'+list(data.columns)[i]] = data.iloc[:,i]**2

# Check
print("Input data shape:")
print(data.shape)
print(list(data.columns))

tmp = pd.merge(data, t1hr_max, left_index=True, right_on='timestamp')
y = tmp['prec']

idx2015 = sum(tmp['timestamp']<=2015010101)
print('Data index before 2015')
print(idx2015)
idx2016 = sum(tmp['timestamp']<=2016010101)
print('Data index before 2016')
print(idx2016)

print("Output data shape:")
print(y.shape)

# Split data
x_train = data.iloc[:idx2015,:]
x_test = data.iloc[idx2015:idx2016,:]
y_train = y.iloc[:idx2015,]
y_test = y.iloc[idx2015:idx2016,]

print("Training data dimension:")
print(x_train.shape)
print(y_train.shape)
print("Test data dimension:")
print(x_test.shape)
print(y_test.shape)

#GLM
import statsmodels.api as sm

# Fit the classifier
glm = sm.GLM(y_train.reset_index(drop=True), x_train.fillna(0.).reset_index(drop=True))
glm_results = glm.fit()

print(glm_results.summary())

# RFECV
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Select input variables
xvars = ['full_pc1', 'full_pc2', 'full_pc3', 'full_pc4', 'full_pc5', 'full_pc6', 'full_pc7', 'full_pc8', 'full_pc9', 'full_pc10', 'fp40_pc1', 'fp40_pc2', 'fp40_pc3', 'fp40_pc4', 'fp40_pc5', 'fp40_pc6', 'fp40_pc7', 'fp40_pc8', 'fp40_pc9', 'fp40_pc10', 'ftyw_pc1', 'ftyw_pc2', 'ftyw_pc3', 'ftyw_pc4', 'ftyw_pc5', 'ftyw_pc6', 'ftyw_pc7', 'ftyw_pc8', 'ftyw_pc9', 'ftyw_pc10', 'squared_full_pc1', 'squared_full_pc2', 'squared_full_pc3', 'squared_full_pc4', 'squared_full_pc5', 'squared_full_pc6', 'squared_full_pc7', 'squared_full_pc8', 'squared_full_pc9', 'squared_full_pc10', 'squared_fp40_pc1', 'squared_fp40_pc2', 'squared_fp40_pc3', 'squared_fp40_pc4', 'squared_fp40_pc5', 'squared_fp40_pc6', 'squared_fp40_pc7', 'squared_fp40_pc8', 'squared_fp40_pc9', 'squared_fp40_pc10', 'squared_ftyw_pc1', 'squared_ftyw_pc2', 'squared_ftyw_pc3', 'squared_ftyw_pc4', 'squared_ftyw_pc5', 'squared_ftyw_pc6', 'squared_ftyw_pc7', 'squared_ftyw_pc8', 'squared_ftyw_pc9', 'squared_ftyw_pc10']
# Build a classification task using 3 informative features
X = x_train.reset_index(drop=True).loc[:,xvars[:10]]
y = y_train.reset_index(drop=True)

print("Starting feature selection:")
print("  input data dimesnion: "+str(X.shape))
print("  output data dimesnion: "+str(y.shape))
# Create the RFE object and compute a cross-validated score.
svc = SVR(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=KFold(5), scoring='neg_mean_squared_error', verbose=1, n_jobs=5)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
joblib.dump(rfecv, 'rfecv_fitted.joblib')

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




