#!/bin/bash
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_full -n 50 -l full.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_fp40 -f ../examples/data/dates_p40.csv -n 50 -l fp40.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_fp20 -f ../examples/data/dates_p20.csv -n 50 -l fp20.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_fp10 -f ../examples/data/dates_p10.csv -n 50 -l fp10.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_fp05 -f ../examples/data/dates_p05.csv -n 50 -l fp05.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_fp01 -f ../examples/data/dates_p01.csv -n 50 -l fp01.log
python ../utils/dbz_ipca_with_filter.py -i ../data/qpesums_lst_hwc/ -o ln_ftyw -f ../examples/data/dates_typhoon.csv -n 50 -l ftyw.log
