#!/bin/bash
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_full -m ln_fp40.pca.joblib -l ln_fp40.log
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_fp40 -m ln_fp40.pca.joblib -l ln_fp40.log
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_fp20 -m ln_fp20.pca.joblib -l ln_fp20.log
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_fp10 -m ln_fp10.pca.joblib -l ln_fp10.log
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_fp05 -m ln_fp05.pca.joblib -l ln_fp05.log
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_fp01 -m ln_fp01.pca.joblib -l ln_fp01.log
python ../utils/dbz_ipca_transform.py -i ../qpesums_lst_hwc/ -o proj_ln_ftyw -m ln_ftyw.pca.joblib -l ln_ftyw.log
