import pyrubberband as pyrb
import soundfile as sf
import librosa

def addpitch(input_file, output_file, pitch_shift):
    # 音声データの読み込み
    audio, sr = librosa.load(input_file,sr=16000)
    # ピッチ変更を適用
    shifted_audio = pyrb.pitch_shift(audio, sr, pitch_shift)
    # 出力ファイルとして保存
    sf.write(output_file, shifted_audio, sr)

sounds = ["M001.wav", "M002.wav", "M003.wav", "M004.wav", "M005.wav"]

for sound in sounds:
    addpitch("sound/"+sound, "sound/P2_"+sound, 2)