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

import model
from data import *

with open("config.json") as f:
  config = json.load(f)

device = torch.device("cpu")

#with open(config["path"]["labelled_list"], "rb") as f:
  #list_scans = pickle.load(f)
list_scans = glob(os.path.join(config["path"]["scans"],"*.mhd"))

st_scans = [os.path.basename(s) for s in list_scans]
print(st_scans[0])

dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode="3d", scan_size=config["train3d"]["scan_size"], n_classes=config["train3d"]["n_classes"])

criterion = utils.lossTang2019 #utils.dice_loss
unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model"))

scanIDs = [i.replace(".mhd","")[-4:] for i in dataset.list_scans]
for i in range(len(dataset)):
    X,y = dataset.__getitem__(i)
    X = torch.Tensor(np.array([X.astype(np.float16)])).to(device) #scan
    y = torch.Tensor(np.array([y.astype(np.float16)])).to(device) #labels
    logits = unet(X)
    mask = logits.cpu().detach().numpy()
    labelMap = torch.max(logits[0],0)[1].numpy()
    im = sitk.GetImageFromArray(labelMap)
    writeName = "./segmentations/mask3D_labelMap"+str(scanIDs[i])+".mhd"
    sitk.WriteImage(im, writeName) #sitk.WriteImage(writeName, labelMap)
    utils.binaryLabelToSTL(writeName,writeName.replace(".mhd", ".stl"))    

