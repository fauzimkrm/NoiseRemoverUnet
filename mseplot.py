import pandas as pd
import matplotlib.pyplot as plt
import os

y_min=0
y_max=0.004
x_min=0
x_max=50

path="Eksperimen/3DATA_FAUZI_SN-5010"

for cuDir,dirs,files in os.walk(path):
    for file in files:
        #print(file)
        if file.endswith(".csv"):
            df_file=os.path.join(cuDir,file)
            df=pd.read_csv(df_file)
            #print(df.tail(10))     #末尾10行
            #print(df.columns)
            df=df[["MSE","val_MSE"]][0:(len(df)-2)]
            df.columns=["train","vali"]
            #print(df)
            plt.figure()
            df.plot()
            plt.ylim(y_min,y_max)
            plt.xlim(x_min,x_max)
            plt.title("MSE")
            plt.savefig(os.path.join(cuDir,"newMSE.png"))
            plt.close("all")

