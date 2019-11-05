#!/bin/bash
STDS=`ls ../data/precipitation.tpe/*.csv`
#for f in $STDS; do
#    id="${i%.*}";
#    python ../utils/qpesums_cnn_reg.py -x ../data/qpesums_lst/ -y $f -o $id.csv -b 64 -e 10 -k 5;
#done
#python ../utils/qpesums_cnn_reg.py -x ../data/qpesums_lst/ -y ../data/1hrmax.csv -o max -b 64 -e 50 -k 3 -l max.log
#python ../utils/qpesums_cnn_reg.py -x ../data/qpesums_lst/ -y ../data/1hrmax.csv -o logmax -g 1 -b 64 -e 50 -k 3 -l logmax.log
python ../utils/qpesums_cnn_mlc.py -x ../data/qpesums_lst/ -y ../data/1hrmax.csv -o max -b 64 -e 10 -k 3 -l max.log
