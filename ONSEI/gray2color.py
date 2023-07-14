import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

path=path = f'image/'

fns = [
  # f'{ss}_in_000.png',
  # f'{ss}_in_001.png',
  # f'{ss}_out_000.png',
  # f'{ss}_out_001.png',
   f'i.png',
   f'l.png',
   f'sabun_l.png',
  # f'sabun_{ss}_out_000.png',
  # f'sabun_{ss}_out_001.png',
]

cmap = plt.get_cmap('jet')
for fn in fns:
  fn = os.path.join(path, fn)
  print(fn)
  img = cv2.imread(fn, 0)*4
  #差分画像時は*2^n倍する(見やすくするための調整)
  print(np.min(img), np.max(img))
  out = cmap(img, bytes=True)
  print(out.shape)
  out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGR)
  print(out.shape)
  # cv2.imshow('out', out)
  # cv2.waitKey()
  cv2.imwrite(fn, out)
# cv2.destroyWindow('out')