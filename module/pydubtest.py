from pydub import AudioSegment
import os



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

#デシベルの変換
#new_sound=sound+10

#無音データの作成
#new_sound=AudioSegment.silent(sound.duration_seconds*1000,frame_rate=16000)

#音声を追加
#new_sound=new_sound.append(sound,crossfade=0)


if __name__=="__main__":
    fileNumber=1
    a=str(fileNumber).zfill(3)
    fname="M"+a
    savefolder="./sound"
    soundCombine(fname,savefolder)
