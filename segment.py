import argparse
from glob import glob
from os import mkdir
from sys import exit

import hjson
import model
import numpy as np
import SimpleITK as sitk
import torch
import vedo as v
from cleanup_tools import CleanupTools
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
    "-c",
    "--config_file",
    default="config.json",
    type=str,
    help="configuration json file /path/to/config.json [default config.json]",
  )
  parser.add_argument(
    "-bd",
    "--bounding_box_dir",
    default="bounding_boxes/",
    type=str,
    help="Directory to save bounding boxes",
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

with open(args.config_file) as f:
  config = hjson.load(f)
segID = config["segment3d"]["output_id"]

device = torch.device("cpu")
#mkdir("segmentations", exist_ok=True)

criterion = utils.lossTang2019  # utils.dice_loss
unet = model.UNet(
  1, config["train3d"]["n_classes"], config["train3d"]["start_filters"]
).to(device)
unet.load_state_dict(torch.load("./unet-model.pt"))

kwargs = {}
try:
  crop_to_lobes = config["segment3d"]["crop_to_lobes"]
except:
  crop_to_lobes = False
if crop_to_lobes:
  kwargs["crop_to_lobes"] = crop_to_lobes
  print(f"Reading lobes file {config['segment3d']['lobes_dir']}")
  lobes_file = glob(config["segment3d"]["lobes_dir"])[0]
  print(f"glob finds: {glob(config['segment3d']['lobes_dir'])}")
  # kwargs["lobe_seg"] = config["segment3d"]["lobes_dir"]
  lobes_seg = sitk.ReadImage(lobes_file)
  lobes_arr = sitk.GetArrayFromImage(lobes_seg)
  lungs_arr = np.where(lobes_arr != 0, 1, 0)
  kwargs["lobe_seg"] = sitk.GetImageFromArray(lungs_arr)
  kwargs["lobe_seg"].CopyInformation(lobes_seg)
  # erode so that it is like cropping to airways
  kwargs["lobe_seg"] = CleanupTools.binary_erode(kwargs["lobe_seg"], 10)
  del lungs_arr, lobes_seg, lobes_arr

dataset = seg_dataset.SegmentSet(
  args.ct_path,
  crop_fraction=config["segment3d"]["crop_fraction"], 
  **kwargs,
)

# get data
(
  X, 
  X_orig,
  bounding_box_to_tissue,
  bounding_box_to_lobes,
) = dataset.__getitem__()

print(f"read data, bounding_box_to_lobes {bounding_box_to_lobes}")

np.savetxt(
  f"{args.bounding_box_dir}/bounding_box_to_tissue-{segID}.txt",
  bounding_box_to_tissue,
  header="xmin, ymin, zmin, xsize, ysize, zsize",
)
np.savetxt(
  f"{args.bounding_box_dir}/bounding_box_to_lobes-{segID}.txt",
  bounding_box_to_lobes,
  header="xmin, ymin, zmin, xsize, ysize, zsize",
)

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
  image_to_write = sitk.GetImageFromArray(segmentation[l].T)
  image_to_write.CopyInformation(X_orig)
  sitk.WriteImage(image_to_write, args.output_surface.replace(".stl", ".mhd"), True)

  vol = v.Volume(
    np.pad(segmentation[l].T, 1),
    spacing=X_orig.GetSpacing(),
    origin=X_orig.GetOrigin(),
  )
  print(f"isosurface spacing {X_orig.GetSpacing()}")
  print("vol to isosurface")
  mesh = vol.isosurface()
  v.write(mesh, args.output_surface)
