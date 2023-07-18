import csv
import argparse
import os
from sys import path
# import keras, cv2, glob
import cv2
import glob
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Dropout,Reshape
from keras.layers import concatenate,Conv2D,BatchNormalization,Activation,Dropout,Reshape
# from keras.optimizers import Adam,SGD,Nadam
from numpy.core.shape_base import block
# from tensorflow.python.keras import Input
from keras import Input
# from tensorflow.keras.models import Model
from keras.models import Model
# from tensorflow.python.keras.backend import binary_crossentropy, conv1d, one_hot, relu, sigmoid,set_session
from keras.backend import binary_crossentropy, conv1d, one_hot, relu, sigmoid,set_session
# from tensorflow.python.keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import UpSampling2D
# from tensorflow.python.keras.layers.pooling import  MaxPooling2D,MaxPool2D
from keras.layers.pooling import  MaxPooling2D,MaxPool2D
# from tensorflow.python.keras.models import load_model
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument('--projectname', '-pn', type=str, default='project')
parser.add_argument('--epoch', '-e', type=int, default=10)
parser.add_argument('--batchsize', '-bs', type=int, default=16)
args = parser.parse_args()

project_name = args.projectname
epochs = args.epoch
batch_sizes = args.batchsize
result_save_dir = "/"+project_name

##pdf化---設定
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
dir="dataset/data_prepocessing_result/images"
path_train_Data=dir+"/out_train/"
path_Noise_Data=dir+"/in_train/"
# path_test_Data=dir+"/in_test/" #通常
path_input_vali=dir+"/in_vali/"
path_output_vali=dir+"/out_vali/"

# #保存先
# os.makedirs(result_save_dir+"/PDF",exist_ok=True)
# os.makedirs(result_save_dir+"/PNG",exist_ok=True)

dir=path_train_Data
image_size=256
X=[] #input_train
Y=[] #output_train
# Z=[] #input_test
A=[] #input_validation
B=[] #output_validation

#Train
files=glob.glob(dir+"*.png")
for i,file in enumerate(files):
    image_gray=cv2.imread(file,0)#グレースケールで読み込み
    image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
    data=np.array(image)#一元化
    X.append(data)#設定
    #print(data.shape)


X=np.array(X)
X= X.astype('float32')
X=X/255.0
#print(X)

dir=path_Noise_Data
files=glob.glob(dir+"*.png")

for i,file in enumerate(files):
    image_gray=cv2.imread(file,0)#グレースケールで読み込み
    image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
    data=np.array(image)#一元化
    Y.append(data)#設定
    #print(data.shape)

Y=np.array(Y)
Y= Y.astype('float32')
Y=Y/255.0

#Test
# dir=path_test_Data
# files=glob.glob(dir+"*.png")

# for i,file in enumerate(files):
#     image_gray=cv2.imread(file,0)#グレースケールで読み込み
#     image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
#     data=np.array(image)#一元化
#     Z.append(data)#設定
#     #print(data.shape)

# Z=np.array(Z)
# Z= Z.astype('float32')
# Z = Z / 255.0

#Validation
dir=path_input_vali
files=glob.glob(dir+"*.png")

for i,file in enumerate(files):
    image_gray=cv2.imread(file,0)#グレースケールで読み込み
    image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
    data=np.array(image)#一元化
    A.append(data)#設定
    #print(data.shape)

A=np.array(A)
A= A.astype('float32')
A = A / 255.0

dir=path_output_vali
files=glob.glob(dir+"*.png")

for i,file in enumerate(files):
    image_gray=cv2.imread(file,0)#グレースケールで読み込み
    image=cv2.resize(image_gray,(image_size,image_size))#リサイズ
    data=np.array(image)#一元化
    B.append(data)#設定
    #print(data.shape)

B=np.array(B)
B= B.astype('float32')
B = B / 255.0


#unet
def blocks(input,filt):
    inputs=input
    for i in range(2):
        inputss=Conv2D(filt,3,padding='same',kernel_initializer='he_normal')(inputs)
        inputss=BatchNormalization()(inputss)
        inputs=Activation('relu')(inputss)
    return inputs

def unet(input=Input(shape=(256,256,1))):
    
    conv1=blocks(input,64)
    x=MaxPooling2D(pool_size=(2,2),padding='same')(conv1)
    
    conv2=blocks(x,128)
    x=MaxPooling2D(pool_size=(2,2),padding='same')(conv2)

    conv3=blocks(x,256)
    x=MaxPooling2D(pool_size=(2,2),padding='same')(conv3)

    conv4=blocks(x,512)
    x=MaxPooling2D(pool_size=(2,2),padding='same')(conv4)

    conv5=blocks(x,1024)
    
    up4=Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    co4=concatenate([conv4,up4],3)
    conv4d=blocks(co4,512)
    
    up3=Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4d))
    co3=concatenate([conv3,up3],3)
    conv3d=blocks(co3,256)
    
    up2=Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3d))
    co2=concatenate([conv2,up2],3)
    conv2d=blocks(co2,128)
    
    up1=Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2d))
    co1=concatenate([conv1,up1],3)
    conv1d=blocks(co1,64)
    
    conv=Conv2D(1,1,activation='sigmoid')(conv1d)
    
    return Model(input,conv)

autoencoder = unet()

# callbacks per 50 epoch
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 50 == 0:  # or save after some epoch, each k-th epoch etc.
            # RENAME THE MODEL IF YOU WANT
            self.model.save(result_save_dir+"/"+project_name+"{}.h5".format(epoch))
saver = CustomSaver()

#訓練設定
# autoencoder.compile(optimizer=Adam(lr=5e-4), loss='mean_squared_error',metrics=['mae','mse','binary_crossentropy'])
adam = keras.optimizers.Adam(lr=5e-4)
autoencoder.compile(optimizer=adam, loss='mean_squared_error',metrics=['mae','mse','binary_crossentropy'])


#訓練実行
print("trainout="+str(len(X)))
print("trainin="+str(len(Y)))
# print("test="+str(len(Z)))
print("valiin="+str(len(A)))
print("valiout="+str(len(B)))

auto_fit=autoencoder.fit(Y,X,
                epochs=epochs,
                batch_size=batch_sizes,
                shuffle=True,
                validation_data=(A,B),
                callbacks=[saver])


#モデル保存
#autoencoder.save("./result/densya/m4096/save_modeldayo.h5")
autoencoder.save(result_save_dir+"/"+project_name)

#各結果history
loss=auto_fit.history['loss']
mae=auto_fit.history['mae']
mse=auto_fit.history['mse']
bicro=auto_fit.history['binary_crossentropy']
val_mae=auto_fit.history['val_mae']
val_mse=auto_fit.history['val_mse']
val_bicro=auto_fit.history['val_binary_crossentropy']


#csv作成
#csv用配列作成
i=0
csvarray=np.empty([epochs+1,8]) #配列枠作成[行,列]

while(i<epochs):
    csvarray[i][0]=i+1
    csvarray[i][1]=loss[i]
    csvarray[i][2]=mae[i]
    csvarray[i][3]=mse[i]
    csvarray[i][4]=val_mae[i]
    csvarray[i][5]=val_mse[i]
    csvarray[i][6]=bicro[i]
    csvarray[i][7]=val_bicro[i]
    i+=1

#csv書き込み
with open(result_save_dir+'/'+project_name+'.csv','w',newline="") as csvfile:
    w=csv.writer(csvfile)
    w.writerow(['Epoch','Loss(Train)','MAE','MSE','val_MAE','val_MSE','binary_crossentropy','val_binary_crossentropy'])
    w.writerows(csvarray)
csvfile.close()


#PDF
pdf=PdfPages(result_save_dir+"/PDF/Loss.pdf")
fig=plt.figure()
ax=fig.add_subplot(1,1,1,xlabel='Epoch',ylabel='Loss',xlim=(0,epochs+1),title="Loss")
ax.plot(range(1,epochs+1),loss,label='Loss')
#ax.suptitle('Loss')
ax.legend()
fig.savefig(result_save_dir+"/PNG/Loss.png")
pdf.savefig(fig)
pdf.close()


pdf=PdfPages(result_save_dir+"/PDF/MAE.pdf")
fig=plt.figure()
ax=fig.add_subplot(1,1,1,xlabel='Epoch',ylabel='Loss',xlim=(0,epochs+1),title="MAE")
ax.plot(range(1,epochs+1),mae,label='train')
ax.plot(range(1,epochs+1),val_mae,label='validation')
ax.legend()
fig.savefig(result_save_dir+"/PNG/MAE.png")
pdf.savefig(fig)
pdf.close()


pdf=PdfPages(result_save_dir+"/PDF/MSE.pdf")
fig=plt.figure()
ax=fig.add_subplot(1,1,1,xlabel='Epoch',ylabel='Loss',xlim=(0,epochs+1),title="MSE")
ax.plot(range(1,epochs+1),mse,label='train')
ax.plot(range(1,epochs+1),val_mse,label='validation')
ax.legend()
fig.savefig(result_save_dir+"/PNG/MSE.png")
pdf.savefig(fig)
pdf.close()


pdf=PdfPages(result_save_dir+"/PDF/BICRO.pdf")
fig=plt.figure()
ax=fig.add_subplot(1,1,1,xlabel='Epoch',ylabel='Loss',xlim=(0,epochs+1),title="binary_crossentropy")
ax.plot(range(1,epochs+1),bicro,label='train')
ax.plot(range(1,epochs+1),val_bicro,label='validation')
ax.legend()
fig.savefig(result_save_dir+"/PNG/BICRO.png")
pdf.savefig(fig)
pdf.close()
