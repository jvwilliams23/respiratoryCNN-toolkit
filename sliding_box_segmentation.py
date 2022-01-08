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
    bounding_box_to_tissue,
    bounding_box_to_lobes,
  ) = dataset.__getitem__(i)
  # save bounding boxes to be used in cleanup
  np.savetxt(
    f"bounding_box_to_tissue-{segID}.txt",
    bounding_box_to_tissue,
    header="xmin, ymin, zmin, xsize, ysize, zsize",
  )
  np.savetxt(
    f"bounding_box_to_lobes-{segID}.txt",
    bounding_box_to_lobes,
    header="xmin, ymin, zmin, xsize, ysize, zsize",
  )
  labelMap_windows = [None] * len(input_windows)
  for w, window in enumerate(input_windows):
    window = torch.Tensor(
      np.array([window[np.newaxis, :].astype(np.float16)])
    ).to(
      device
    )  # scan
    print("-" * 30, "Getting probabilities")
    logits = unet(window)
    print("-" * 30, "Getting label map")
    labelMap_windows[w] = torch.max(logits[0], 0)[1].numpy()
    del logits
  vol_list = []
  box_size = (upper_list[0] - lower_list[0]) // 2
  print(f"box size is {box_size}")
  combined_vol = copy(labelMap_windows[0])
  for j, (window, origin) in enumerate(
    zip(labelMap_windows[1:], origin_list[1:])
  ):
    print(
      j,
      mid_point_list[j],
      window[:, :, :box_size].shape,
      combined_vol[:, :, mid_point_list[j] :].shape,
    )
    # Calculate union of two windows of a binary label map
    combined_vol[:, :, mid_point_list[j] :] += window[:, :, :box_size]
    combined_vol[:, :, mid_point_list[j] :] = np.where(
      combined_vol[:, :, mid_point_list[j] :] >= 1, 1, 0
    )
    print("\t", combined_vol.shape[-1], window[:, :, box_size:].shape[-1])
    combined_vol = np.dstack((combined_vol, window[:, :, box_size:]))
    print("\t", combined_vol.shape[-1])
  image_out = sitk.GetImageFromArray(combined_vol)
  image_out.SetSpacing(spacing)
  image_out.SetOrigin(origin_list[0])
  print("Writing labelMap to mhd")
  sitk.WriteImage(
    image_out, "./segmentations/seg-{0}-{1}.mhd".format(segID, "airway")
  )
  print("numpy to volume")
  vol = v.Volume(combined_vol, spacing=spacing)
  # mesh = vol.isosurface()#largest=True)
  mesh = vol.isosurface(largest=True)
  print("Writing vtk surface mesh")
  v.write(mesh, f"{segID}_mm_airway.vtk")
