from pydub import AudioSegment



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

if __name__=="__main__":
    fname="alarm clock.wav"
    ncopy(fname)