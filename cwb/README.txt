# Deep Learning Based Volume-to-Point QPE
# 使用說明

## 1. 安裝
1. 將程式資料夾複製到欲執行工作的主機上
2. 確認主機上已安裝：
  - python 3.6 以上
  - numpy
  - pandas
  - h5py
  - tensorflow 2.0 以上 (or tensorflow-gpu)


## 2. 程式結構
  /
  主目錄，包含主程式

  /model
  預先訓練好的模型

  /qpesums_source
  預設的 QPESUMS 文字檔資料夾

  /qpesums_lst_hwc
  預設的 QPESUMS 二進位檔資料夾


## 3. 資料需求
1. QPESUMS 台灣地區資料，純文字格式，內容為平面網格上的最大回波強度。

2. 檔案內部格式為「固定寬度文字」，三欄位分別為長度8個字元的「緯度」、「經度」、及「回波強度」 (ny=275, nx=162)

3. 檔名格式為 "COMPREF.YYYYmmdd.HHMM.txt"，範例：COMPREF.20160101.0010.txt

4. 每整點以之前60分鐘的六筆資料疊加後進行 QPE，如欲缺檔則無法進行。例如：0800LST 的 QPE 需要 0700, 0710, 0720 0730, 0740, 0750 六個時間的資料。


## 4. 使用範例
1. 將 QPESUMS 文字檔轉換為二進位檔
[語法]
  > python cwb_qpesums_preprocessing.py -i [輸入：QPESUMS 文字檔資料夾] -o [輸出：QPESUMS 二進位檔資料夾]

[範例]
  > python cwb_qpesums_preprocessing.py -i qpesums_source/ -o qpesums_lst_hwc/



2. 使用二進位的 QPESUMS  資料以及訓練好的模型進行 QPE
[語法]
  > python cwb_qpesums_predict.py -i [輸入：QPESUMS 二進位檔] -o [輸出：預測結果]
[範例]
  > python cwb_qpesums_predict.py -i qpesums_lst_hwc\2016071515.npy -o 2016071515.csv


3. 預報結果判讀：
預報輸出為 .csv 格式，包含：
    id: 測站代號，TPEALL 為整個台北地區
    y0~y4: 參考用
    prediction: QPE 結果是否大於 30mm/hr

