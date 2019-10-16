#!/bin/bash
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_full --log_flag 1 -l ln_full.log -e 10
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp40 -f ../examples/data/dates_p40.csv --log_flag 1 -l ln_fp40.log -e 500
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp20 -f ../examples/data/dates_p20.csv --log_flag 1 -l ln_fp20.log -e 100
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp10 -f ../examples/data/dates_p10.csv --log_flag 1 -l ln_fp10.log -e 100
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp05 -f ../examples/data/dates_p05.csv --log_flag 1 -l ln_fp05.log -e 50
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_fp01 -f ../examples/data/dates_p01.csv --log_flag 1 -l ln_fp01.log -e 10
python ../utils/dbz_conv_autoencoder.py -i ../data/qpesums_npy/ -o ln_ftyw -f ../examples/data/dates_typhoon.csv --log_flag 1 -l ln_ftyw.log -e 100
