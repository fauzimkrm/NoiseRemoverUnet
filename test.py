# import csv
# from sys import path
import cv2
# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import glob
import os
import module.data_prepocessing_tool as tool
import librosa
import soundfile as sf
import shutil
# from PIL import Image
# from keras.layers import concatenate
#from keras.optimizers import Adam,SGD,Nadam
# from tensorflow.keras.optimizers import Adam,SGD,Nadam
# from numpy.core.shape_base import block
# from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Dropout,Reshape
# from tensorflow.python.keras import Input
# from tensorflow.keras.models import Model
# from tensorflow.python.keras.backend import binary_crossentropy, conv1d, one_hot, relu, sigmoid,set_session
# from tensorflow.python.keras.layers.convolutional import UpSampling2D
# from tensorflow.python.keras.layers.pooling import  MaxPooling2D,MaxPool2D
# from keras.models import load_model
# from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument('--inputpath', '-ip', type=str, help='Input path')
parser.add_argument('--outputname', '-on', type=str, help='Output name')
parser.add_argument('--model', '-on', type=str, help='Model path')
args = parser.parse_args()

input_path = args.inputpath
output_name = args.outputname
model_path = args.model
os.makedirs('dataset/temp/test/data',exist_ok=True)
os.makedirs('dataset/temp/test/label',exist_ok=True)
os.makedirs('dataset/temp/test/result',exist_ok=True)

# ==================== PREPROCESS DATA TEST =======================
# data prepocessing variable
datadir = 'dataset/temp/test/data'
labeldir = 'dataset/temp/test/label'
dirs = {'data' : datadir, 'label' : labeldir}
sr = 0.8
nfft = 512

def kansuu(dwav,lwav):
  # パラメータ設定
  ## 入力する2つのwavファイル
  wavs = {'data' : dwav, 'label' : lwav}

  # wavファイルを読み込み，STFTに変換
  stft = {}
  for key, wav in wavs.items():
    _, stft[key] = tool.snd2stft(wav, nfft=nfft)

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
    min_max, buf = tool.stft2dB(stft_dat=dat)
    # dBから画像に変換
    imgs[key] = tool.dB2img(buf, alpha=alpha)
  #cv2.imwrite('wavimg.png', imgs['data'])
  #cv2.imwrite('labimg.png', imgs['label'])

  # 画像を切り出して保存
  sw = int(h * sr) # 切出し位置のスライド幅

  #ファイルを置き換えせず、新しいファイルを追加するためのコード
  dir1=datadir      #dataフォルダのパス
  fileName=glob.glob(dir1+'/*.png')    #dataフォルダの中のファイル名を取得
  # print(fileName)        #最後のファイル名を確認する
  sumOfFile=(sum(os.path.isfile(os.path.join(dir1,name)) for name in os.listdir(dir1)))   #フォルダ内のファイル数を数える
  if(sumOfFile==0):        #フォルダが空いている場合
    nlfn=-(sw)
  else:
    lastFileName=fileName[-1]        #最後のファイル名を取得
    nlfn=lastFileName[-11:-4]       #最後のファイル名の数字の部分を取り出す

#   print(nlfn)
  for key, img in imgs.items():
    dir = dirs[key]
    if not os.path.isdir(dir): # ディレクトリが無ければ作る
      os.mkdir(dir)
    for i in range(0, w - h, sw): # iが切り出し位置
      #fname = f'{str(i+int(nlfn)+sw).zfill(7)}.png' # 切り出し位置がファイル名
      fname = f'{str(i+int(nlfn)+sw).zfill(7)}.png'
      cv2.imwrite(os.path.join(dir, fname), img[:, i:i+h])

kansuu(input_path, input_path)

# ================= PREDICT ========================
# predict variable
image_size=256
dir="./result/densya" #door/timer　番号切り替え
longdata="/SN10_P_50epoch_bs4_sr08"
input="/input/"
test_data = 'dataset/temp/test/data'
files1=glob.glob(test_data+"*.png")
inputIMG=[]

#モデルの読み込み
model=tf.keras.models.load_model(model_path)

#サマリー確認
#print(model.summary())

files=sorted(files1)#整列

for i,file in enumerate(files):
    image_gray=cv2.imread(file,0)
    image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
    data=np.array(image)#一元化
    inputIMG.append(data)#設定
    #print(data.shape)

inputIMG=np.array(inputIMG)
inputIMG= inputIMG.astype('float32')
inputIMG = inputIMG / 255.0

print("inputIMG="+str(len(inputIMG)))

#予測
batch_sizes=5
pred=model.predict(inputIMG,batch_size=5)
i=0

#保存
pathresult= 'dataset/temp/test/result/'
i=0
while(i<int(len(inputIMG))):
    a=np.squeeze(pred[i])
    a=a*255.0
    a=a.astype('uint8')
    #数値を4桁に揃える
    zero1=205*i
    zero=str(zero1).zfill(7)
    cv2.imwrite(pathresult+zero+".png",a)
    #print(a.shape)
    i+=1

# ====================== IMG2WAV =========================
# パラメータ設定
outwav = output_name
imgdir = 'dataset/temp/test/result'
sr = 16000

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
dB = tool.img2dB(catimg, alpha=alpha)
# dBデータをSTFTに逆変換
stft = librosa.db_to_amplitude(dB)
# 音データに逆変換(パラメータはwav2img.pyとそろえる)(位相推定)
isnd = librosa.griffinlim(stft, hop_length=128, win_length=512)

# wavファイルに保存
sf.write(outwav, isnd, sr, format='WAV', subtype='PCM_16')
