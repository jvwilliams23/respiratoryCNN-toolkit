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

dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode="3d", n_classes=config["train3d"]["n_classes"])

criterion = utils.dice_loss
unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model"))

for i in range(len(dataset)):
    X,y = dataset.__getitem__(i)
    X = torch.Tensor(np.array([X.astype(np.float16)])).to(device)
    y = torch.Tensor(np.array([y.astype(np.float16)])).to(device)
    logits = unet(X)
    loss = criterion(logits, y)
    mask = logits.cpu().detach().numpy()
    '''
    for label in range(mask.shape[1]):
        nrrd.write("mask3D"+str(label)+".nrrd", mask[0][label])
    nrrd.write("mask3D"+".nrrd", mask[0])
    '''
    labelMap = torch.max(logits,1)[1][0]#.numpy()
    print("old dice coeff",loss)
    print("new dice coeff",utils.compute_dice_coefficient(labelMap, y[0][0], np.arange(1,5)))

