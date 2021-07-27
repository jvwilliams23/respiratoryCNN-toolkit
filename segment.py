import numpy as np
import pickle, nrrd, json, os
import torch
import SimpleITK as sitk
from data.seg_dataset import resampleImage
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from glob import glob
from sys import exit
import nrrd
from os import mkdir

import model
from data import *

with open("config.json") as f:
  config = json.load(f)

device = torch.device("cpu")

if len(glob('segmentations/')) == 0:
    mkdir('segmentations')

#list_scans = glob(config["path"]["test_scans"])
list_scans = glob(config["path"]["test_scans"]) #-Lists all .mhd scans.

#st_scans = [os.path.basename(s) for s in list_scans]

print("st_scans", list_scans)



#dataset = seg_dataset.SegmentSet(list_scans, config["path"]["test_scans"], mode="3d", scan_size=config["train3d"]["scan_size"])

print(config["path"]["test_scans"])

criterion = utils.lossTang2019 #utils.dice_loss
unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model.pt"))

writeNRRDlib = False
writeSITK = True
for i in range(len(list_scans)):
    dataset = seg_dataset.SegmentSet(list_scans[i], list_scans[i], mode="3d", scan_size=config["train3d"]["scan_size"])
    segID = str(list_scans[i].replace(".mhd", ""))[-4:]
    if 'EXACT09' in list_scans[i]:
        segID = 'CASE'+list_scans[i].split('CASE')[1][:2]
    print("-"*30,"Getting and pre-processing scan")
    print('writeID is', segID)
    X, X_orig = dataset.__getitem__(i)
    X = torch.Tensor(np.array([X.astype(np.float16)])).to(device) #scan
    print("-"*30, "Getting probabilities")
    logits = unet(X)
    print("-"*30, "Getting label map")
    labelMap = torch.max(logits[0],0)[1].numpy()
    labelMap = np.swapaxes(labelMap, 0, 2)
    print("-"*30, "Getting largest island for each label")
    segmentation = [None]*len([1])
    for l, label in enumerate([1]):
        segmentation[l] = utils.getLargestIsland(labelMap==label)
        segmentation[l] = sitk.GetImageFromArray(np.array(segmentation[l],dtype=np.int8))
        segmentation[l] = sitk.GetArrayFromImage(segmentation[l])
        segmentation[l] = np.swapaxes(segmentation[l], 1, 2)
        segmentation[l] = np.swapaxes(segmentation[l], 0, 2)
        segmentation[l] = np.swapaxes(segmentation[l], 1, 0)
        print("-"*30, "Upsampling label map", label)
        segmentation[l] = sitk.GetImageFromArray(segmentation[l])
        segmentation[l] = resampleImage(segmentation[l], X_orig.GetSize(), interpolator=sitk.sitkNearestNeighbor)
        segmentation[l].SetOrigin(X_orig.GetOrigin())
        segmentation[l].SetSpacing(X_orig.GetSpacing())
        print("-"*30, "Writing label map", label)
        sitk.WriteImage(segmentation[l], "./segmentations/seg-{0}-{1}.mhd".format(segID, "aw"))
        utils.binaryLabelToSTL("./segmentations/seg-{0}-{1}.mhd".format(segID, "aw"),
                    "./segmentations/{0}_mm_{1}.stl".format(segID, "aw"))

#spacing = [str(X_orig.GetSpacing()[0]), str(X_orig.GetSpacing()[1]), str(X_orig.GetSpacing()[2])]
#centre =  ['"'+str(X_orig.GetOrigin()[0])+'"', '"'+str(X_orig.GetOrigin()[1])+'"', '"'+str(X_orig.GetOrigin()[2])+'"']
#header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['mm', 'mm', 'mm'], 'spacings': spacing, 'space': 'right-anterior-superior', 'centerings':centre}
#nrrd.write("mask3D_segBin.nrrd", labelArr, header, compression_level=9)

