# https://www.kaggle.com/laavanya/elderly-fall-prediction-and-detection?select=readme.docx
import csv
import torch
filename= "Data/cStick.csv"
f = open(filename,mode='r',encoding='utf-8')
reader = csv.reader(f) #读取f赋值给reader对象
with open("Data/cStick.txt", mode='w', encoding='utf8') as ftxt:
    for item in reader:
        if(item[1]=="Pressure"):
            continue
        ftxt.write(" ".join([str(x) for x in item]))
        ftxt.write("\n")