import argparse
import os

import cv2
import numpy as np

import cvtsndimg

import glob

# 実行時引数，オプションの設定
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', '-dd', type=str, default='data', help='データ画像出力ディレクトリ')
parser.add_argument('--labeldir', '-ld', type=str, default='label', help='ラベル画像出力ディレクトリ')
parser.add_argument('--nfft', '-nf', type=int, default=512, help='FFTのデータ長')
parser.add_argument('--sliderate', '-sr', type=float, default=0.8, help='切出し位置のスライド率(0.0<sr<1.0)')

args = parser.parse_args()
#print(args.datawav)     #datawavのファイル名
#print(args.labelwav)    #labelwavのファイル名
def kansuu(dwav,lwav):
  # パラメータ設定
  ## 入力する2つのwavファイル
  wavs = {'data' : dwav, 'label' : lwav}
  ## 出力ディレクトリ
  dirs = {'data' : args.datadir, 'label' : args.labeldir}
  nfft = args.nfft
  sr = args.sliderate

  # wavファイルを読み込み，STFTに変換
  stft = {}
  for key, wav in wavs.items():
    _, stft[key] = cvtsndimg.snd2stft(wav, nfft=nfft)

  # データのサイズを取得
  #print(stft['data'].shape)
  h = stft['data'].shape[0]
  w = np.min([stft['data'].shape[1], stft['label'].shape[1]])     #np.min()最小値
  print(h, w)

  # STFTデータを画像に変換
  isFlip = False # 画像を上下反転する(True)，しない(False)
  alpha = 100. # 画像化するときの画素値への加算値
  imgs = {}
  for key, dat in stft.items():
    # STFTからdBに変換
    min_max, buf = cvtsndimg.stft2dB(stft_dat=dat)
    # dBから画像に変換
    imgs[key] = cvtsndimg.dB2img(buf, alpha=alpha)
  #cv2.imwrite('wavimg.png', imgs['data'])
  #cv2.imwrite('labimg.png', imgs['label'])

  # 画像を切り出して保存
  sw = int(h * sr) # 切出し位置のスライド幅

  #ファイルを置き換えせず、新しいファイルを追加するためのコード
  dir1=args.datadir      #dataフォルダのパス
  fileName=glob.glob(dir1+'\*.png')    #dataフォルダの中のファイル名を取得
  print(fileName)        #最後のファイル名を確認する
  sumOfFile=(sum(os.path.isfile(os.path.join(dir1,name)) for name in os.listdir(dir1)))   #フォルダ内のファイル数を数える
  if(sumOfFile==0):        #フォルダが空いている場合
    nlfn=-(sw)
  else:
    lastFileName=fileName[-1]        #最後のファイル名を取得

    nlfn=lastFileName[5:12]       #最後のファイル名の数字の部分を取り出す
    print(nlfn)

  #print(sw)
  for key, img in imgs.items():
    dir = dirs[key]
    if not os.path.isdir(dir): # ディレクトリが無ければ作る
      os.mkdir(dir)
    for i in range(0, w - h, sw): # iが切り出し位置
      #fname = f'{str(i+int(nlfn)+sw).zfill(7)}.png' # 切り出し位置がファイル名
      fname = f'{str(i+int(nlfn)+sw).zfill(7)}SN10_P0.png'
      cv2.imwrite(os.path.join(dir, fname), img[:, i:i+h])