import csv
from sys import path
import keras, cv2, glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import concatenate
#from keras.optimizers import Adam,SGD,Nadam
from tensorflow.keras.optimizers import Adam,SGD,Nadam
from numpy.core.shape_base import block
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Dropout,Reshape
from tensorflow.python.keras import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.backend import binary_crossentropy, conv1d, one_hot, relu, sigmoid,set_session
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.pooling import  MaxPooling2D,MaxPool2D
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages




#入力データ読み込み
image_size=256
dir="./result/densya" #door/timer　番号切り替え
longdata="/SN10_P_50epoch_bs4_sr08"
input="/input/"
files1=glob.glob(dir+longdata+input+"*.png")
inputIMG=[]


#モデルの読み込み
model=tf.keras.models.load_model(dir+longdata+"/SN10_P_50epoch_bs4_sr08.h5")

#サマリー確認
#print(model.summary())

files=sorted(files1)#整列

for i,file in enumerate(files):
    image_gray=cv2.imread(file,0)
    image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
    data=np.array(image)#一元化
    inputIMG.append(data)#設定
    #print(data.shape)

inputIMG=np.array(inputIMG)
inputIMG= inputIMG.astype('float32')
inputIMG = inputIMG / 255.0

print("inputIMG="+str(len(inputIMG)))


#予測
batch_sizes=5
pred=model.predict(inputIMG,batch_size=5)
i=0

#保存

output="/output/"
pathresult=dir+longdata+output
i=0
while(i<int(len(inputIMG))):
    a=np.squeeze(pred[i])
    a=a*255.0
    a=a.astype('uint8')
    #数値を4桁に揃える
    zero1=205*i
    zero=str(zero1).zfill(7)
    cv2.imwrite(pathresult+zero+".png",a)
    #print(a.shape)
    i+=1


