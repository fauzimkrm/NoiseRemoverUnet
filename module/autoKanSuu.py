import os
import argparse
import glob
import random
import math
import shutil

# parser = argparse.ArgumentParser()
# parser.add_argument('--datadir', '-dd', type=str, default='dataset/temp/data', help='データ画像ディレクトリ')
# parser.add_argument('--labeldir', '-ld', type=str, default='dataset/temp/label', help='ラベル画像ディレクトリ')
# parser.add_argument('--in_validir', '-ivd', type=str, default='dataset/data_prepocessing_result/images/in_vali', help='in_vali画像ディレクトリ')
# parser.add_argument('--in_traindir', '-itd', type=str, default='dataset/data_prepocessing_result/images/in_train', help='in_train画像ディレクトリ')
# parser.add_argument('--out_validir', '-ovd', type=str, default='dataset/data_prepocessing_result/images/out_vali', help='out_vali画像ディレクトリ')
# parser.add_argument('--out_traindir', '-otd', type=str, default='dataset/data_prepocessing_result/images/out_train', help='out_train画像ディレクトリ')
# parser.add_argument('--noisySoundDir', '-nsd', type=str, default='sound/noisySound', help='ノイズありの音声ディレクトリ')
# parser.add_argument('--cleanSoundDir', '-csd', type=str, default='sound/cleanSound', help='ノイズなしの音声ディレクトリ')
# args = parser.parse_args()


datadir='dataset/temp/data'
labeldir='dataset/temp/label'
in_train='dataset/data_prepocessing_result/images/in_train'
out_train='dataset/data_prepocessing_result/images/out_train'
in_vali='dataset/data_prepocessing_result/images/in_vali'
out_vali='dataset/data_prepocessing_result/images/out_vali'

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
