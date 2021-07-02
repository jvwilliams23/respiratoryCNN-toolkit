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
#val_scans and train_scansi
list_scans= glob(os.path.join(config["path"]["unseen_scans"],"*.mhd"))
st_scans=[os.path.basename(s) for s in list_scans]
test_scans=st_scans[:config["train3d"]["test_size"]]
test_data = dataset.Dataset(test_scans, config["path"]["unseen_scans"], config["path"]["unseen_masks"], mode = "3d", scan_size = config["train3d"]["scan_size"], n_classes = config["train3d"]["n_classes"])
for i in test_scans:
  print(i)

criterion = utils.dice #utils.dice_loss
unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model.pt"))
print(len(test_data))
#testList=[list(i) for i in test_data]
for i in range(len(test_data)):
    losses = np.zeros(config["train3d"]["n_classes"]-1) # -1 as we are not interested in background nowi
    print(test_scans[i])
    header=test_scans[i].replace(".mhd","").replace("EXACT09_09","")+"t "
    print(header)
    X,y=test_data.__getitem__(i)
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
dataframe.to_csv("resultsTest.csv")
rocdf=pd.DataFrame.from_dict(ROCdata,orient="columns")
rocdf.to_csv("rocresultsTest.csv")
