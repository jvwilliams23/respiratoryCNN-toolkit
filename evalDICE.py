import numpy as np
import pickle, nrrd, json, os
import torch
import SimpleITK as sitk
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from glob import glob
from sys import exit
import myshow as ms
from sklearn.metrics import roc_curve,auc,confusion_matrix,fbeta_score
import model
from data import *
import re
import pandas as pd
with open("config.json") as f:
  config = json.load(f)

device = torch.device("cpu")
resultdata={}
ROCdata={}
#with open(config["path"]["labelled_list"], "rb") as f:
  #list_scans = pickle.load(f)
#val_scans and train_scans
#dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode="3d", n_classes=config["train3d"]["n_classes"])
def getvaltrain(key):
  scans=[]
  gather=False
  trainscan=re.compile("train scans:.*?")
  valscan=re.compile("validation scans:.*?")
  datagetend=re.compile("train_data OK.*?")
  logFile="logTrain"
  if key =='v':
    with open(logFile) as file:
      for line in file:
        if trainscan.match(line):
          gather=False
          continue
        if valscan.match(line):
          gather=True
          continue
        if datagetend.match(line):
          return scans
        if gather==True:
          scans.append(line.replace('\n',''))
  elif key == 't':
    with open(logFile) as file:
      for line in file:
        if trainscan.match(line):
          gather=True
          continue
        if valscan.match(line):
          gather=False
          continue
        if datagetend.match(line):
          return scans
        if gather==True:
          scans.append(line.replace('\n',''))
  return(scans)
val_scans=getvaltrain('v')
train_scans=getvaltrain('t')
train_data = dataset.Dataset(train_scans, config["path"]["scans"], config["path"]["masks"], mode="3d", scan_size = config["train3d"]["scan_size"], n_classes = config["train3d"]["n_classes"])
val_data = dataset.Dataset(val_scans, config["path"]["scans"], config["path"]["masks"], mode = "3d", scan_size = config["train3d"]["scan_size"], n_classes = config["train3d"]["n_classes"])
for i in val_scans:
  print(i)

criterion = utils.dice #utils.dice_loss
unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model.pt"))
print(len(train_data)+len(val_data))
for i in range(len(train_data)+len(val_data)):
    losses = np.zeros(config["train3d"]["n_classes"]-1) # -1 as we are not interested in background now
    if i < len(train_data):
      X,y = train_data.__getitem__(i)
      #f.write("t"+str(st_scans[i].replace(".mhd", "")[-4:] ))
      header=train_scans[i].replace(".mhd","")[-4:]+"t "
    else:
      X,y = val_data.__getitem__(i-len(train_data))
      header=val_scans[i-len(train_data)].replace(".mhd","")[-4:]+"v "
      #f.write("v"+str(st_scans[i].replace(".mhd", "")[-4:] ))
    #X,y = dataset.__getitem__(i)
    X = torch.Tensor(np.array([X.astype(np.float16)])).to(device)
    y = torch.Tensor(np.array([y.astype(np.float16)])).to(device)
    logits = unet(X)
    #ms.myshow(sitk.GetImageFromArray(np.array(X[0][0])))
    #ms.myshow(sitk.GetImageFromArray(np.array(torch.nn.functional.softmax(logits, dim=1)[0][1].detach().numpy())))
    #exit()
    #f.write(str(st_scans[i].replace(".mhd", "")[-4:] ))
    """
    for c, channel in enumerate(losses): 
      losses[c] = criterion(torch.max(logits, dim=1)[0][c+1], y,c+1)
      f.write("\t"+str(round(losses[c],4)))
    f.write("\n")
    """
    #y_pred_bin=F.softmax(logits,dim=1).detach().numpy().ravel()
    #y_true=y.detach().numpy().ravel()
    #y_pred_bin=np.argmax(logits.detach().numpy(),axis=-1)
    y_pred=torch.argmax(logits,1)
    tn,fp,fn,tp=confusion_matrix(y.view(-1),y_pred.view(-1)).ravel()
    dc=2*tp/((tp+fp)+(tp+fn))
    fpr,tpr,_=roc_curve(y.view(-1),y_pred.view(-1))
    roc_auc=auc(fpr,tpr)
    fbeta=fbeta_score(y.view(-1),y_pred.view(-1),beta=0.5)
    resultdata.update({header+"DC":[dc]})
    resultdata.update({header+"AUC":[roc_auc]})
    ROCdata.update({header+"FPR":fpr.tolist()})
    ROCdata.update({header+"TPR":tpr.tolist()})
    resultdata.update({header+"fbeta":[fbeta]})
#save data to csv file
print(resultdata)
dataframe=pd.DataFrame.from_dict(resultdata,orient="columns")
dataframe.to_csv("results.csv")
rocdf=pd.DataFrame.from_dict(ROCdata,orient="columns")
rocdf.to_csv("rocresults.csv")
