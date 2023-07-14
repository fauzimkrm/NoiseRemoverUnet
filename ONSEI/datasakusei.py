#学習データ作成
import wave
import argparse
import array
import numpy as np
import math
import random
from pydub import AudioSegment
import os

import wav2img
import autoKanSuu
import SNtest
import pydubtest1
import noisecopy

parser = argparse.ArgumentParser()
parser.add_argument('--sounddir', '-snd', type=str, default='sound', help='音声ディレクトリ')
parser.add_argument('--noisysounddir', '-nsnd', type=str, default='noisysound', help='重畳音声出力ディレクトリ')
args = parser.parse_args()


init=1
end=5
snr=10
ratio=0.9
pitch=""


nname="TrainingNoise"
t=".wav"


#学習データセットを格納するフォルダの作成
autoKanSuu.makeDirs()

#ノイズデータが〇秒未満の場合、複製する
noisesound=AudioSegment.from_wav("sound/"+nname+t)
if(noisesound.duration_seconds<600):
    noisecopy.ncopy(nname+t)

# for i in range(init,end+1): #init番目の音声からend番目の音声まで
#     a=str(i).zfill(3) #1の場合は001になる
#     fname="M"+a
#     #短い音声を繋げる、2つ目の引数：音声の保存フォルダ、3つ目の引数：音声の長さ指定
#     pydubtest1.soundCombine(fname,args.sounddir)
#     print(f"{fname}{t}")



for i in range(init,end+1):
    #ノイズ重畳
    fnumber=str(i).zfill(3)
    # fname="P"+pitch+"_M"+fnumber
    fname="M"+fnumber
    print(fname)
    #音声を読み込む
    cwavfile=wave.open(args.sounddir+"/"+fname+t)
    #ノイズを読み込む
    nwavfile=wave.open(args.sounddir+"/"+nname+t)
    #SN比でノイズの振幅調整
    camp,crms=SNtest.getamp_rms(cwavfile)
    namp,nrms=SNtest.getamp_rms(nwavfile)
    print("Clean : ", camp.shape)
    print("Noise : ", namp.shape)
    nrms_new=SNtest.snr_cal(crms,snr)
    #音声と同じ長さのノイズを切り出す
    start=random.randint(0,len(namp)-len(camp))
    namp_new=namp[start:start+len(camp)]
    #音声の振幅とSN比指定した後の振幅を足し合わせ
    namp_new=namp_new*(nrms_new/nrms)
    mixed_amp=(camp+namp_new)

    #足し合わせた振幅の最大値が2の15乗を超えた場合、正規化する
    if (mixed_amp.max(axis=0)>32767):
        mixed_amp=mixed_amp*(32767/mixed_amp.max(axis=0))
        camp=camp*(32767/mixed_amp.max(axis=0))
        namp_new=namp_new*(32767/mixed_amp.max(axis=0))
    #ノイズ重畳音声の保存
    noisy_wav=wave.Wave_write(args.noisysounddir+"/"+fname+nname+"SN"+str(snr)+t)
    noisy_wav.setparams(cwavfile.getparams())
    noisy_wav.writeframes(array.array('h', mixed_amp.astype(np.int16)))
    noisy_wav.close()

    print(fname+nname+"SN"+str(snr)+t)
    #ノイズなしとノイズ重畳の音声を画像に変換
    wav2img.kansuu(args.sounddir+"/"+fname+t,args.noisysounddir+"/"+fname+nname+"SN"+str(snr)+t)

#画像を訓練用データと評価用データにランダムで分割する
autoKanSuu.IMG_split(ratio)
print("終了") 