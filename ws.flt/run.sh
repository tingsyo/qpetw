#!/bin/bash
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_npy/ -o fp40 -f ../examples/data/dates_p40.csv -n 100 -l fp40.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_npy/ -o fp20 -f ../examples/data/dates_p20.csv -n 100 -l fp20.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_npy/ -o fp10 -f ../examples/data/dates_p10.csv -n 100 -l fp10.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_npy/ -o fp05 -f ../examples/data/dates_p05.csv -n 100 -l fp05.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_npy/ -o fp01 -f ../examples/data/dates_p01.csv -n 100 -l fp01.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_npy/ -o tyw -f ../examples/data/dates_typhoon.csv -n 100 -l tyw.log
