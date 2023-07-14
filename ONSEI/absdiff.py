import os
import sys

import cv2
import numpy as np

# python absdiff.py [file1] [file2] で実行
# file1, file2は比較する2つのファイル(画像)
# file1とfile2は同じ拡張子

if len(sys.argv) == 3:
  # ファイル名取得
  file1 = sys.argv[1]
  file2 = sys.argv[2]
  # 拡張子取得
  ext1 = os.path.splitext(file1)[1]
  ext2 = os.path.splitext(file2)[1]
  if ext1 == ext2:
    img_gray1 = cv2.imread(file1, flags=0)
    img_gray2 = cv2.imread(file2, flags=0)

    img1 = cv2.resize(img_gray1,(256,256))
    img2 = cv2.resize(img_gray2,(256,256))

    img_diff = cv2.absdiff(img1, img2)
    cv2.imwrite('sabun_'+file2,img_diff)
    
    
  else:
    print('2つのファイルの拡張子は揃える')
else:
  print('引数に比較する2つのファイル(csvか画像)を指定')



'''
img_1 = cv2.imread('sabuntest1.jpg',0)
img_2 = cv2.imread('sabuntest2.jpg',0)

img_diff = cv2.absdiff(img_1, img_2)

cv2.imwrite('sabun-result.jpg',img_diff)
'''