#!/bin/bash
STDS=`ls ../data/precipitation.tpe/*.csv`
for f in STDS; do
    id="${i%.*}";
    python ../utils/qpesums_cnn_reg.py -x ../data/qpesums_lst/ -y ../data/precipitation.tpe/$f -o $id.csv -b 128 -e 100 -k 10;
done
