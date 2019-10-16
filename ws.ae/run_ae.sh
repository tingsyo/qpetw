#!/bin/bash
#python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_full -n 100 -l full.log
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp40 -f ../examples/data/dates_p40.csv --log_flag 1 -l fp40.log
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp20 -f ../examples/data/dates_p20.csv --log_flag 1 -l fp20.log
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp10 -f ../examples/data/dates_p10.csv --log_flag 1 -l fp10.log
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp05 -f ../examples/data/dates_p05.csv --log_flag 1 -l fp05.log
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp01 -f ../examples/data/dates_p01.csv --log_flag 1 -l fp01.log
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_ftyw -f ../examples/data/dates_typhoon.csv --log_flag 1 -l ftyw.log
