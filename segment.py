import argparse
from os import mkdir
from sys import exit

import hjson
import model
import numpy as np
import SimpleITK as sitk
import torch
import vedo as v
from data import *


def inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "-i",
    "--ct_path",
    required=True,
    type=str,
    help="input image to segment /path/to/ct/*.mhd or /path/to/dicom/",
  )
  parser.add_argument(
    "-o",
    "--output_surface",
    default="seg-1.stl",
    type=str,
    help="Airway segmentation surface file name",
  )

  return parser.parse_args()


args = inputs()

with open("config.json") as f:
  config = hjson.load(f)

device = torch.device("cpu")
#mkdir("segmentations", exist_ok=True)

criterion = utils.lossTang2019  # utils.dice_loss
unet = model.UNet(
  1, config["train3d"]["n_classes"], config["train3d"]["start_filters"]
).to(device)
unet.load_state_dict(torch.load("./unet-model.pt"))

dataset = seg_dataset.SegmentSet(
  args.ct_path,
  crop_fraction=config["segment3d"]["crop_fraction"],
)

# get data item i
X, X_orig = dataset.__getitem__()

# format data to be read by NN
X = torch.Tensor(np.array([X.astype(np.float16)])).to(device)  # scan
logits = unet(X)
labelMap = torch.max(logits[0], 0)[1].numpy()
labelMap = np.swapaxes(labelMap, 0, 2)

segmentation = [None] * len([1])
for l, label in enumerate([1]):
  segmentation[l] = utils.extract_largest_island(labelMap)

  # get metadata from original ct dataset
  # segmentation[l].SetOrigin(X_orig.GetOrigin())
  # segmentation[l].SetSpacing(X_orig.GetSpacing())
  # add extra layer of voxels to make closed surface
  # pad = sitk.ConstantPadImageFilter()
  # pad.SetPadLowerBound((1, 1, 1))
  # pad.SetPadUpperBound((1, 1, 1))
  # pad.SetConstant(0)
  # segmentation[l] = pad.Execute(segmentation[l])
  sitk.WriteImage(sitk.GetImageFromArray(segmentation[l]), args.output_surface.replace(".stl", ".mhd"), True)

  vol = v.Volume(
    np.pad(segmentation[l].T, 1),
    spacing=X_orig.GetSpacing(),
    origin=X_orig.GetOrigin(),
  )
  print(f"isosurface spacing {X_orig.GetSpacing()}")
  print("vol to isosurface")
  mesh = vol.isosurface()
  v.write(mesh, args.output_surface)
