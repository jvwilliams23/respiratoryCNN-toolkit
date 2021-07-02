import numpy as np
import pickle, json, os
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from glob import glob
import SimpleITK as sitk

import model
from data import *
from myshow import *
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
###############################
#25/11/2020
#additions: random_val
import random
import re
from statistics import mean
from collections import defaultdict
##############################
from sys import exit
torch.backends.cudnn.benchmark = True

#-Open and load configurations.
with open("config.json") as f:
  config = json.load(f)

device = torch.device("cpu")
#with open(config["path"]["scans"], "rb") as f:
#list_scans = pickle.load(f)
list_scans = glob(os.path.join(config["path"]["scans"],"*.mhd")) #-Lists all .mhd scans.
st_scans = [os.path.basename(s) for s in list_scans]  #get all scans
test=True
#randomly select scan paths set in the config file, the other are left for training only 
#this avoids repetitive human error
def random_val(All_scans):
    sys_Random=random.SystemRandom()
    n=config["train3d"]["validation_size"]
    if test == True:
      testScans_=[]
      for i in All_scans:
        if i.find("EXACT09") != -1:
          index = All_scans.index(i)
          testScans_.append(All_scans.pop(index))
      sys_Random.shuffle(All_scans)
      #trainingScans=Allscans[10:]+testScans_
      return All_scans[:10],All_scans[10:]+testScans_
    else:
      sys_Random.shuffle(All_scans)
      return All_scans[:10],All_scans[10:]    
def stop_training():
    logdata=[]
    logdata_index=[]
    validationsection=False
    iterator=0
    #pattern = re.compile("Epoch [0-9]{1,3} ==> Batch (.*?)")
    #patterndata = re.compile("DICE IS tensor(.*?)")
    patternDICE = re.compile("DICE IS tensor(.*?)(\d*$)(.*?)")
    validationbegin=re.compile("(.*?)Calculating validation loss(.*?)")
    validationend=re.compile("(.*?)Validation Loss(.*?)")
    #value = (lambda match: match.group(1) if match else '')(re.search(regex,text))
    with open("logTrain") as file:
        for line in file:
            if validationbegin.match(line):
                validationsection=True
                continue
            if validationend.match(line):
                iterator=iterator+1
                validationsection=False
                continue
            if validationsection==True:
                #if epochnum.match(line):
                    #logdata_index.append(int(epochnum.match(line).group(1)))
                    #add bool to only keep airway data
                #    continue
                if patternDICE.match(line):
                    logdata.append(float((re.sub('[^\d\.]','',line))))
                    logdata_index.append(iterator)
                    continue
    result = []
    temp = defaultdict(list)
    #need a dict(epoch number,[list of values])
    
    logdata=list(zip(logdata[1::2],logdata_index[1::2]))
    
    for ele in logdata: 
        temp[ele[1]].append(ele[0])
    
    for k,v in temp.items():
        result.append((np.average(np.array(v))).tolist())
    #delete the unnecessary
    xs=np.arange(start=1, stop=30,step=1)
    ys=np.array(result[len(result)-29:])
    gradient=(lambda xs,ys: ((mean(xs)*mean(ys)) - mean(xs*ys)) /((mean(xs)**2) - mean(xs**2)))(xs,ys)
    if gradient>-0.0016:
        print("gradient = ",gradient,"exiting")
        exit(0)

if config["mode"] == "3d":
  print("Running 3D")
  try:
    All_scans = st_scans[:]
    val_scans,train_scans=random_val(All_scans)
  except IndexError as error:
    print(error, " size of container:",str(len(list_scans))," size of index: ",str(config["train3d"]["train_size"]+config["train3d"]["validation_size"]))
    exit(0)
  print("train scans: ")
  for a in train_scans:
    print(a)
  print("validation scans: ")
  for a in val_scans:
    print(a)
  train_data = dataset.Dataset(train_scans, config["path"]["scans"], config["path"]["masks"], mode="3d", scan_size = config["train3d"]["scan_size"], n_classes = config["train3d"]["n_classes"])
  print("train_data OK. Number of scans: ",len(train_data))
  #list_scans = train_scans #scans_path = config["path"]["scans"] #masks_path = config["path"]["masks"]
  val_data = dataset.Dataset(val_scans, config["path"]["scans"], config["path"]["masks"], mode = "3d", scan_size = config["train3d"]["scan_size"], n_classes = config["train3d"]["n_classes"])
  unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"], bilinear = False).to(device)
  criterion = utils.lossTang2019 #utils.dice_loss_old #utils.compute_dice_coefficient #utils.dice_loss
  optimizer = optim.Adam(unet.parameters(), lr = config["train3d"]["lr"])
  batch_size = config["train3d"]["batch_size"]
  epochs = config["train3d"]["epochs"]
  val_steps = config["train3d"]["validation_steps"]
  val_size = config["train3d"]["validation_size"]
  print("End of data adquisition")
else:
  print("Running 2D")
  st_scans = st_scans[:config["train2d"]["train_size"]]
  #dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode = "2d")
  train_data = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode = "2d")
  unet = model.UNet(1,1, config["train2d"]["start_filters"], bilinear = True).to(device)
  criterion = utils.dice_loss
  optimizer = optim.Adam(unet.parameters(), lr = config["train2d"]["lr"])
  batch_size = config["train2d"]["batch_size"]
  slices_per_batch = config["train2d"]["slices_per_batch"]
  neg = config["train2d"]["neg_examples_per_batch"]
  epochs = config["train2d"]["epochs"]
#crop image
#def crop_image(image):
    
# augmentation pipeline for images
aug_images = iaa.Sequential([
    iaa.Rotate((-15.0,15.0)),
    iaa.AdditiveGaussianNoise(scale=(0.0,0.5))
])#alpha is representative of initial displacement, sigma is representative of smoothing strength

# augmentation pipeline for segmentation maps - with coarse dropout, but without gaussian noise
aug_segmaps = iaa.Sequential([
    iaa.Rotate((-15.0,15.0))
])

aug_iters = config["train3d"]["aug_iters"] 
unet.train()
best_val_loss = 1e16
for epoch in range(epochs):
  if epoch > 30:
    stop_training()
  epoch_loss = 0
  for i in range(0, len(train_data)+aug_iters, batch_size):
    losses = torch.ones(config["train3d"]["n_classes"])
    batch_loss = 0
    if i >= len(train_data):
       augID = np.random.randint(len(train_data)-batch_size-1) #-pick random ID to augment 
       print("Augmenting patient", augID)
       batch = np.array([train_data.__getitem__(j)[0] for j in range(augID, augID+batch_size)]).astype(np.float16)
       labels = np.array([train_data.__getitem__(j)[1] for j in range(augID, augID+batch_size)]).astype(np.int8)
       #augment segmentation and scan using the same random state
       aug_images = aug_images.localize_random_state()
       aug_segmaps = aug_segmaps.localize_random_state()
       aug_segmaps_= aug_segmaps.to_deterministic()
       aug_images_= aug_images.to_deterministic()
       aug_segmaps_=aug_segmaps_.copy_random_state(aug_segmaps)
       batch[0][0] = aug_images_.augment_image(batch[0][0])
       labels[0][0] = aug_segmaps_.augment_image(labels[0][0])
    else:
       batch = np.array([train_data.__getitem__(j)[0] for j in range(i, i+batch_size)]).astype(np.float16)
       labels = np.array([train_data.__getitem__(j)[1] for j in range(i, i+batch_size)]).astype(np.int8) 
    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)
    batch.requires_grad = True
    labels.requires_grad = True
    #print("CHECK", torch.sum(labels==0), torch.sum(labels==1), torch.sum(labels==2))
    #exit()
    optimizer.zero_grad()
    logits = unet(batch)
    print(type(logits))
    #loss = criterion(torch.max(logits,1)[1].numpy()[0], labels.detach().numpy()[0][0])
    for c, channel in enumerate(losses): 
      losses[c] = criterion(torch.nn.functional.softmax(logits, dim=1)[0][c], labels,c)
    print("LOSSES ARE", losses)
    
    #print("CHECK", torch.sum(labels==0), torch.sum(labels==1), torch.sum(labels==2))
    #exit()
    
    loss = torch.mean(losses)
    loss.backward()#retain_graph=True)
    optimizer.step()
    print("Epoch {} ==> Batch {} mean loss : {}".format(epoch+1, (i+1)%(val_steps), loss.item()/batch_size))
    epoch_loss += loss/batch_size
    del batch
    del labels
    del logits
    torch.cuda.empty_cache()
    print("val_steps =",val_steps)
    
    #if (i+1)%val_steps == 0:
    if i==len(train_data)-1:
      print("===================> Calculating validation loss ... ")
      ids = np.random.randint(0, len(val_data), val_size)
      print("ids", ids)
      val_loss = 0
      loop=0
      for scan_id in ids:
        print("Number of ids. ", ids)
        losses = torch.ones(config["train3d"]["n_classes"])
        batch = np.array([val_data.__getitem__(j)[0] for j in range(scan_id, scan_id+batch_size)]).astype(np.float16)
        #print("val_data batch\n", batch)
        labels = np.array([val_data.__getitem__(j)[1] for j in range(scan_id, scan_id+batch_size)]).astype(np.int8)
        #print("val_data labels\n", labels)

        batch = torch.Tensor(batch).to(device)
        labels = torch.Tensor(labels).to(device)
        loop+=1
        print("Loop no. ", loop)
        logits = unet(batch)
        for c, channel in enumerate(losses):
          losses[c] = criterion(torch.nn.functional.softmax(logits, dim=1)[0][c], labels,c)
          #losses[c] = criterion(torch.max(logits,1)[1][0], labels[0][0], c+1
        loss = torch.mean(losses)
        #loss = criterion(torch.max(logits,1)[1][0], labels[0][0], np.arange(1,6))
        #loss = criterion(torch.max(logits,1)[1].numpy()[0], labels.detach().numpy()[0][0])
        #loss = criterion(logits, labels)
        val_loss += loss.item()
        del batch 
        del labels
        del logits
        del losses
      val_loss /= val_size
      print("\n # Validation Loss : ", val_loss)
      if val_loss < best_val_loss:
        print("\nSaving Better Model... ")
        torch.save(unet.state_dict(), "./model.pt")
        best_val_loss = val_loss
      print("\n")
