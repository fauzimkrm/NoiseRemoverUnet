from pydub import AudioSegment

fname="Futamatagawa-Yokohama_40s.wav"

def ncopy(fname):
    newfname="new"+fname
    sound=AudioSegment.from_wav("noisysound/"+fname)

    print(f"音声の長さ：{sound.duration_seconds}秒")
    new_sound=AudioSegment.empty()
    while(new_sound.duration_seconds<1200):
        new_sound=new_sound.append(sound,crossfade=0)

    print(f"音声の長さ：{new_sound.duration_seconds}秒")
    new_sound.export("./noisysound//"+fname,format="wav")

if __name__=="__main__":
        ncopy(fname)