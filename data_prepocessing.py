#学習データ作成
import wave
import argparse
import array
import numpy as np
import math
import random
from pydub import AudioSegment
import os

import module.wav2img as wav2img
import module.autoKanSuu as autoKanSuu
import module.SNtest as SNtest
import module.pydubtest as pydubtest
import module.noisecopy as noisecopy

parser = argparse.ArgumentParser()
parser.add_argument('--sounddir', '-sd', type=str, default='dataset/sound/clear', help='音声ディレクトリ')
parser.add_argument('--noisedir', '-nd', type=str, default='dataset/sound/noise', help='音声ディレクトリ')
parser.add_argument('--snr', '-snr', type=str, default=0, help='Signal noise ration')
parser.add_argument('--holdout', '-ho', type=str, default=0.8, help='Holdout validation method for dataset')
parser.add_argument('--pitchswitch', '-ps', type=str, default=0, help='Pitch switch')
parser.add_argument('--noisefilename', '-nfn', type=str, default="TrainingNoise", help='Noise file name')
parser.add_argument('--noisysounddir', '-nsnd', type=str, default='dataset/data_prepocessing_result/mixedsound', help='重畳音声出力ディレクトリ')

args = parser.parse_args()

sound_dir = args.sounddir
noise_dir = args.noisedir
snr = args.snr
ratio = args.holdout
pitch= args.pitchswitch
nname= args.noisefilename
t=".wav"


#学習データセットを格納するフォルダの作成
autoKanSuu.makeDirs()

#ノイズデータが〇秒未満の場合、複製する
noisesound=AudioSegment.from_wav(noise_dir+"/"+nname+t)
if(noisesound.duration_seconds<600):
    noisecopy.ncopy(nname+t)

for fname in os.listdir(sound_dir):
    audio, sr = librosa.load(input_file,sr=16000)
    # ピッチ変更を適用
    shifted_audio = pyrb.pitch_shift(audio, sr, pitch_shift)

    #音声を読み込む
    cwavfile=wave.open(sound_dir+"/"+fname)
    #ノイズを読み込む
    nwavfile=wave.open(noise_dir+"/"+nname+t)
    #SN比でノイズの振幅調整
    camp,crms=SNtest.getamp_rms(cwavfile)
    namp,nrms=SNtest.getamp_rms(nwavfile)
    print("Clean Voice Amplitude Length : ", camp.shape)
    print("Noise Voice Amplitude Length : ", namp.shape)
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
    noisy_wav=wave.Wave_write(args.noisysounddir+"/"+fname[:-4]+nname+"SN"+str(snr)+t)
    noisy_wav.setparams(cwavfile.getparams())
    noisy_wav.writeframes(array.array('h', mixed_amp.astype(np.int16)))
    noisy_wav.close()

    print(fname+nname+"SN"+str(snr)+t)
    #ノイズなしとノイズ重畳の音声を画像に変換
    wav2img.kansuu(args.sounddir+"/"+fname,args.noisysounddir+"/"+fname[:-4]+nname+"SN"+str(snr)+t)

#画像を訓練用データと評価用データにランダムで分割する
autoKanSuu.IMG_split(ratio)
print("終了") 