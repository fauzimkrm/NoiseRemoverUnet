import wave
import argparse
import array
import numpy as np
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument('--sounddir', '-snd', type=str, default='dataset/sound', help='音声ディレクトリ')
parser.add_argument('--noisysounddir', '-nsnd', type=str, default='dataset/data_prepocessing_result/mixedsound', help='重畳音声出力ディレクトリ')
args = parser.parse_args()

#wavファイルから振幅情報(amptitude)と振幅の二乗平均平方根(rms)を取得する
def getamp_rms(wavfile):
    
    buffer=wavfile.readframes(wavfile.getnframes())
    amptitude=(np.frombuffer(buffer,dtype="int16")).astype(np.float64)
    rms=np.sqrt(np.mean(np.square(amptitude),axis=-1))
    return amptitude,rms

#音声データの振幅の二乗平均平方根(crms)とSN比(snr)からノイズの振幅の二乗平均平方根を計算する
def snr_cal(crms,snr):
    a=float(snr)/20
    nrms=crms/(10**a)
    return nrms

if __name__=="__main__":
    snr=0                         #SN比の変更
    fname="M100"                  #元音声
    nname="S5"                  #雑音
    t=".wav"

    #wavファイルを読み込む
    cwavfile=wave.open(args.sounddir+"/"+fname+t)
    nwavfile=wave.open(args.sounddir+"/"+nname+t)

    camp,crms=getamp_rms(cwavfile)
    print(f"音声のポイント数：{len(camp)},振幅の二乗平均平方根：{crms}")
    namp,nrms=getamp_rms(nwavfile)
    print(f"ノイズのポイント数：{len(namp)},振幅の二乗平均平方根：{nrms}")
    nrms_new=snr_cal(crms,snr)
    print(f"SN比指定した後のノイズの振幅の二乗平均平方根：{nrms_new}")
    #音声と同じ長さのノイズを切り出す
    start=random.randint(0,len(namp)-len(camp))
    #start=0
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