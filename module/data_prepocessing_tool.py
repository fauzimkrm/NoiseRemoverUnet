import os
import argparse
import glob
import random
import math
import shutil
import cv2
import librosa
import numpy as np
import scipy.signal as sp
import wave
import argparse
import array
import glob

from pydub import AudioSegment

# ============== Constanta ===================
datadir='dataset/temp/data'
labeldir='dataset/temp/label'
in_train='dataset/data_prepocessing_result/images/in_train'
out_train='dataset/data_prepocessing_result/images/out_train'
in_vali='dataset/data_prepocessing_result/images/in_vali'
out_vali='dataset/data_prepocessing_result/images/out_vali'
sounddir = 'dataset/sound'
noisysounddir = 'dataset/data_prepocessing_result/mixedsound'
datadir = 'dataset/temp/data'
labeldir = 'dataset/temp/label'
nfft = 512
sliderate = 0.8

# =============== Function ===================
# AutoKanSuu
#画像分割
def IMG_split(n):    #引数：訓練用データの割合
    dataIMG=glob.glob(datadir+'/*.png')
    labelIMG=glob.glob(labeldir+'/*.png')
    #訓練用データを乱数で決める
    ransuu=[]
    a=math.floor(len(dataIMG)*n)
    while len(ransuu)<a:
        n=random.randint(0,len(dataIMG)-1)
        if not n in ransuu:
            ransuu.append(n)
    #訓練用データを移動する
    out_trainIMG=[]
    [out_trainIMG.append(dataIMG[i])for i in ransuu]
    [shutil.move(i,out_train)for i in out_trainIMG]
    in_trainIMG=[]
    [in_trainIMG.append(labelIMG[i])for i in ransuu]
    [shutil.move(i,in_train)for i in in_trainIMG]
    #残りのデータを評価用データにする
    [shutil.move(i,in_vali)for i in glob.glob(labeldir+'/*.png')]
    [shutil.move(i,out_vali)for i in glob.glob(datadir+'/*.png')]
    
def makeDirs():
    os.makedirs(in_train,exist_ok=True)
    os.makedirs(out_train,exist_ok=True)
    os.makedirs(in_vali,exist_ok=True)
    os.makedirs(out_vali,exist_ok=True)

# cvtsndimg
# サウンドデータをSTFTに変換する
def snd2stft(sound_file, window='hann', nperseg=512, noverlap=128, nfft=512):
  data, sr = librosa.load(sound_file, sr=None) # サウンドデータの読み込み(
  stft_dat = librosa.stft(data, n_fft=nfft, hop_length=noverlap, win_length=nperseg)
  #n_fft:窓の長さ hop_length:窓の移動幅　win_length:窓関数の中でも実際に窓関数を適用させたい長さ
  return np.shape(stft_dat), stft_dat

# STFTをdB(デシベル)に変換
def stft2dB(stft_dat):
  stft_amp = np.abs(stft_dat) # STFTの絶対値を取って大きさに変換
  stft_dB = librosa.amplitude_to_db(stft_amp)
  return (np.min(stft_dB), np.max(stft_dB)), stft_dB

# デシベルデータを画像に変換
def dB2img(dB, alpha=0., flip=True):
  dB += alpha # alphaはデータをかさ上げする値
  dB[dB < 0] = 0 # 負の値を0にする
  img = dB.astype(np.uint8) # 型変換
  if flip: # 上下反転
    img = cv2.flip(img, 0)
  return img 

# 画像をデシベルデータに変換
def img2dB(img, alpha=0., flip=True):
  if flip: # 上下反転
    img = cv2.flip(img, 0)
  dB = img.astype(np.float32) # 型変換
  dB -= alpha # かさ上げを元に戻す
  return dB

# noisecopy
def ncopy(fname):
    newfname="new"+fname
    sound=AudioSegment.from_wav("sound/"+fname)

    print(f"音声の長さ：{sound.duration_seconds}秒")
    new_sound=AudioSegment.empty()
    while(new_sound.duration_seconds<1200):
        new_sound=new_sound.append(sound,crossfade=0)
        new_sound=new_sound[:1200*1000]

    print(f"音声の長さ：{new_sound.duration_seconds}秒")
    new_sound.export(".\sound\\"+fname,format="wav")

# Combining Sound in Corpus File
def soundCombine(fname,savefolder,duration=-1):
    for cuDir,dirs,files in os.walk("./corpus/"+fname+"/NP"):
        new_sound=AudioSegment.empty()
        #読み込み
        for file in files:
            sound=AudioSegment.from_wav(os.path.join(cuDir,file))
            #音声を繋げる
            new_sound=new_sound.append(sound,crossfade=0)
        #音声ファイルの保存
        if duration!=-1 and duration<=new_sound.duration_seconds:
            new_sound=new_sound[0:duration*1000]                                #print(os.path.join(cuDir,file))
            #print(f"{cuDir}/{file}")
        t=".wav"
        print(f"サンプリング周波数：{new_sound.frame_rate}Hz")
        new_sound.export(savefolder+"/"+fname+t,format="wav")


        if __name__=="__main__":
            print(f"ファイル名：{fname}{t}")
            print(f"音声の長さ：{new_sound.duration_seconds}秒")
            print(f"チャンネル数：{new_sound.channels}")
            print(f"量子化ビット数：{new_sound.sample_width*8}")

# SNTest
#wavファイルから振幅情報(amptitude)と振幅の二乗平均平方根(rms)を取得する
def getamp_rms(wavfile):
    
    buffer=wavfile.readframes(wavfile.getnframes())
    amptitude=(np.frombuffer(buffer,dtype="int16")).astype(np.float64)
    rms=np.sqrt(np.mean(np.square(amptitude),axis=-1))
    return amptitude,rms

# WAV2IMG
#音声データの振幅の二乗平均平方根(crms)とSN比(snr)からノイズの振幅の二乗平均平方根を計算する
def snr_cal(crms,snr):
    a=float(snr)/20
    nrms=crms/(10**a)
    return nrms

def addpitch(input_file, output_file, pitch_shift):
    # 音声データの読み込み
    audio, sr = librosa.load(input_file,sr=16000)
    # ピッチ変更を適用
    shifted_audio = pyrb.pitch_shift(audio, sr, pitch_shift)
    # 出力ファイルとして保存
    sf.write(output_file, shifted_audio, sr)

def kansuu(dwav,lwav, snr, pitch):
  # パラメータ設定
  ## 入力する2つのwavファイル
  wavs = {'data' : dwav, 'label' : lwav}
  ## 出力ディレクトリ
  dirs = {'data' : datadir, 'label' : labeldir}
  sr = sliderate

  # wavファイルを読み込み，STFTに変換
  stft = {}
  for key, wav in wavs.items():
    _, stft[key] = snd2stft(wav, nfft=nfft)

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
    min_max, buf = stft2dB(stft_dat=dat)
    # dBから画像に変換
    imgs[key] = dB2img(buf, alpha=alpha)
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
      fname = 'SNR'+str(snr)+'_PITCH'+str(pitch)+'_'+f'{str(i+int(nlfn)+sw).zfill(7)}.png'
      cv2.imwrite(os.path.join(dir, fname), img[:, i:i+h])
