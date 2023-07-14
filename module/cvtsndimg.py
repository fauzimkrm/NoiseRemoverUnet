import cv2
import librosa
import numpy as np
import scipy.signal as sp

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