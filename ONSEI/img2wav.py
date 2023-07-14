import argparse
import os
from glob import glob        #画像を音に戻す

import cv2
import librosa
import numpy as np
import soundfile as sf

import cvtsndimg

# 実行時引数，オプションの設定
parser = argparse.ArgumentParser()
parser.add_argument('outwav', help='出力wavファイル')
parser.add_argument('--imgdir', '-id', type=str, default='resultIMG', help='結合画像ディレクトリ')
parser.add_argument('--nfft', '-nf', type=int, default=512, help='FFTのデータ長')
parser.add_argument('--smprate', '-sr', type=int, default=16000, help='出力wavサンプリングレート')

args = parser.parse_args()

# パラメータ設定
outwav = args.outwav
imgdir = args.imgdir
nfft = args.nfft
sr = args.smprate

# 画像ファイルリスト作成
imgs = np.sort(glob(os.path.join(imgdir, '*.png')))

# 最初の画像を読み込み，連結用画像に貼り付ける
img = cv2.imread(imgs[0], 0) # 階調画像で読み込み
trim = 5
imgsize = (nfft // 2 + 1, nfft // 2 + 1) # 出力に必要な画像サイズ
if img.shape != imgsize: # 出力に必要なサイズにリサイズ
  img = cv2.resize(img, imgsize, interpolation=cv2.INTER_LANCZOS4)
# 2番目のファイル名を取得(切り出し時のスライド幅を取得する)
x = int(os.path.splitext(os.path.basename(imgs[1]))[0])
# 貼り付け用画像
catimg = np.zeros((imgsize[0], x * len(imgs) + imgsize[1] - x), np.uint8)
catimg[:, :imgsize[1]] = np.copy(img) # 1枚目を貼り付け
for path in imgs[1:]: # 2枚目以降を貼り付け
  x = int(os.path.splitext(os.path.basename(path))[0])
  img = cv2.imread(path, 0)
  if img.shape != imgsize: # 出力に必要なサイズにリサイズ
    img = cv2.resize(img, imgsize, interpolation=cv2.INTER_LANCZOS4)
  catimg[:, x+trim:x+imgsize[1]] = np.copy(img[:, trim:])
cv2.imwrite('catimg.png', catimg)

# 画像をdBデータに逆変換
isFlip = False # wav2img.pyの値とそろえる
alpha = 100.   # wav2img.pyの値とそろえる
dB = cvtsndimg.img2dB(catimg, alpha=alpha)
# dBデータをSTFTに逆変換
stft = librosa.db_to_amplitude(dB)
# 音データに逆変換(パラメータはwav2img.pyとそろえる)(位相推定)
isnd = librosa.griffinlim(stft, hop_length=128, win_length=512)

# wavファイルに保存
sf.write(outwav, isnd, sr, format='WAV', subtype='PCM_16')
