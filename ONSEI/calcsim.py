import os
import sys

import cv2
import numpy as np

# python calssim.py [file1] [file2] で実行
# file1, file2は比較する2つのファイル(csvか画像)
# file1とfile2は同じ拡張子

if len(sys.argv) == 3:
  # ファイル名取得
  file1 = sys.argv[1]
  file2 = sys.argv[2]
  # 拡張子取得
  ext1 = os.path.splitext(file1)[1]
  ext2 = os.path.splitext(file2)[1]
  if ext1 == ext2:
    if ext1 == '.csv' or ext1 == '.CSV':
      # CSVファイルの読み込み
      buf = np.genfromtxt(sys.argv[1], delimiter=',')
      dat1 = buf.astype(np.float32)
      buf = np.genfromtxt(sys.argv[2], delimiter=',')
      dat2 = buf.astype(np.float32)
    else: # csvファイルで無ければ画像とみなす
      dat1 = cv2.imread(file1, flags=0)
      dat2 = cv2.imread(file2, flags=0)
    # 類似度計算
    method=cv2.TM_SQDIFF_NORMED
    #method = cv2.TM_CCORR_NORMED # 正規化相互相関[-1, 1]
    result = cv2.matchTemplate(dat1, dat2, method)
    # 最大類似度の取得
    _, max_val, _, _ = cv2.minMaxLoc(result)
    print(max_val)
  else:
    print('2つのファイルの拡張子は揃える')
else:
  print('引数に比較する2つのファイル(csvか画像)を指定')