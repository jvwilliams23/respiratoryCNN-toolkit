"""
Split image in half with plane in X-direction. Segment each half with UNet then merge outputs

"""

import numpy as np
import pickle
import nrrd
from copy import copy
import hjson
import os
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
from distutils.util import strtobool
import vedo as v

import model
from data import *

with open("config.json") as f:
  config = hjson.load(f)

device = torch.device("cpu")

if len(glob("segmentations/")) == 0:
  mkdir("segmentations")

# list_scans = glob(config["path"]["test_scans"])
list_scans = glob(config["path"]["test_scans"])  # -Lists all .mhd scans.

# st_scans = [os.path.basename(s) for s in list_scans]

print("st_scans", list_scans)


# dataset = seg_dataset.SegmentSet(list_scans, config["path"]["test_scans"], mode="3d", scan_size=config["train3d"]["scan_size"])

print(config["path"]["test_scans"])

criterion = utils.lossTang2019  # utils.dice_loss
unet = model.UNet(
  1, config["train3d"]["n_classes"], config["train3d"]["start_filters"]
).to(device)
unet.load_state_dict(torch.load("./model.pt"))

writeNRRDlib = False
writeSITK = True
kwargs = {}
for i in range(len(list_scans)):
  downsampling_on = strtobool(config["segment3d"]["downsampling_on"][0])
  if downsampling_on:
    downsampling_ratio = config["segment3d"]["downsample"]
    kwargs["downsampling_ratio"] = downsampling_ratio
  kwargs["num_boxes"] = config["segment3d"]["num_boxes"]

  dataset = seg_half_dataset.SegmentSet(
    list_scans[i], list_scans[i], downsample=downsampling_on, **kwargs
  )
  segID = config["path"]["output_id"]
  print("-" * 30, "Getting and pre-processing scan")
  print("writeID is", segID)
  # get data item i
  # X, X_orig = dataset.__getitem__(i)
  (
    input_windows,
    origin_list,
    spacing,
    lower_list,
    mid_point_list,
    upper_list,
  ) = dataset.__getitem__(i)
  labelMap_windows = [None] * len(input_windows)
  for w, window in enumerate(input_windows):
    window = torch.Tensor(np.array([window[np.newaxis,:].astype(np.float16)])).to(device)  # scan
    print("-" * 30, "Getting probabilities")
    logits = unet(window)
    print("-" * 30, "Getting label map")
    labelMap_windows[w] = torch.max(logits[0], 0)[1].numpy()
    del logits
    # labelMap_windows[w] = np.swapaxes(labelMap_windows[w], 0, 2)
  vol_list = []
  box_size = (upper_list[0] - lower_list[0]) // 2
  print(f"box size is {box_size}")
  combined_vol = copy(labelMap_windows[0])
  for j, (window, origin) in enumerate(zip(labelMap_windows[1:], origin_list[1:])):
    print(
      j,
      mid_point_list[j],
      window[:, :, :box_size].shape,
      combined_vol[:, :, mid_point_list[j] :].shape,
    )
    # Calculate intersection of two windows of a binary labelmap 
    #combined_vol[:, :, mid_point_list[j] :] *= window[:, :, :box_size]
    # Calculate union of two windows of a binary label map
    combined_vol[:, :, mid_point_list[j] :] += window[:, :, :box_size]
    combined_vol[:, :, mid_point_list[j] :] = np.where(combined_vol[:, :, mid_point_list[j] :] >= 1, 1, 0)
    print(
      "\t", combined_vol.shape[-1], window[:, :, box_size:].shape[-1]
    )
    combined_vol = np.dstack((combined_vol, window[:, :, box_size:]))
    print("\t", combined_vol.shape[-1])
    # vol_list.append(v.Volume(window.T, spacing=spacing, origin=origin))
  # outputs = vol_list[0].append(vol_list[1:])
  image_out = sitk.GetImageFromArray(combined_vol)
  image_out.SetSpacing(spacing)
  image_out.SetOrigin(origin_list[0])
  print("Writing labelMap to mhd")
  """sitk.WriteImage(
    image_out, "./segmentations/seg-{0}-{1}.mhd".format(segID, "airway")
  )"""
  print("numpy to volume")
  vol = v.Volume(combined_vol, spacing=spacing)
  #mesh = vol.isosurface()#largest=True)
  mesh = vol.isosurface(largest=True)
  print("Writing vtk surface mesh")
  v.write(mesh, f"{segID}_mm_airway.vtk")
  # format data to be read by NN
  """
  labelMap_windows = [None] * len(input_windows)
  for h, half in enumerate(input_windows):
    half = torch.Tensor(np.array([half.astype(np.float16)])).to(device)  # scan
    print("-" * 30, "Getting probabilities")
    logits = unet(half)
    print("-" * 30, "Getting label map")
    labelMap_half[h] = torch.max(logits[0], 0)[1].numpy()
    labelMap_half[h] = np.swapaxes(labelMap_half[h], 0, 2)
  labelMap = np.dstack(labelMap_half)
  print("-" * 30, "Getting largest island for each label")
  segmentation = [None] * len([1])
  for l, label in enumerate([1]):
    segmentation[l] = utils.getLargestIsland(labelMap == label)
    #segmentation[l] = (labelMap == label)
    # change shape to correct orientation for comparing with ground truths
    segmentation[l] = sitk.GetImageFromArray(
      np.array(segmentation[l], dtype=np.int8)
    )
    segmentation[l] = sitk.GetArrayFromImage(segmentation[l])
    segmentation[l] = np.swapaxes(segmentation[l], 1, 2)
    segmentation[l] = np.swapaxes(segmentation[l], 0, 2)
    segmentation[l] = np.swapaxes(segmentation[l], 1, 0)
    print("-" * 30, "Upsampling label map", label)
    segmentation[l] = sitk.GetImageFromArray(segmentation[l])
    segmentation[l] = resampleImage(
      segmentation[l], X_orig.GetSize(), interpolator=sitk.sitkNearestNeighbor
    )
    # get metadata from original ct dataset
    segmentation[l].SetOrigin(X_orig.GetOrigin())
    segmentation[l].SetSpacing(X_orig.GetSpacing())
    # add extra layer of voxels to make closed surface
    pad = sitk.ConstantPadImageFilter()
    pad.SetPadLowerBound((1, 1, 1))
    pad.SetPadUpperBound((1, 1, 1))
    pad.SetConstant(0)
    segmentation[l] = pad.Execute(segmentation[l])
    print("-" * 30, "Writing label map", label)
    # output data
    mesh = utils.numpy_to_surface(
      sitk.GetArrayFromImage(segmentation[l]),
      origin=X_orig.GetOrigin(),
      spacing=spacing,
    )
    v.write(mesh, f"segmentations/{segID}_mm_airway.vtk")
  """


# spacing = [str(X_orig.GetSpacing()[0]), str(X_orig.GetSpacing()[1]), str(X_orig.GetSpacing()[2])]
# centre =  ['"'+str(X_orig.GetOrigin()[0])+'"', '"'+str(X_orig.GetOrigin()[1])+'"', '"'+str(X_orig.GetOrigin()[2])+'"']
# header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['mm', 'mm', 'mm'], 'spacings': spacing, 'space': 'right-anterior-superior', 'centerings':centre}
# nrrd.write("mask3D_segBin.nrrd", labelArr, header, compression_level=9)
