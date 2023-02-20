"""
cnn of lung lobes
"""
import argparse
from glob import glob
from sys import exit
import os

import numpy as np
import torch
import SimpleITK as sitk
from torch.utils import data
from lungmask import mask
import vedo as v

from data import *


def inputs():
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-i",
    "--ct_path",
    required=True,
    type=str,
    help="input image to segment /path/to/ct/*.mhd or /path/to/dicom/",
  )
  parser.add_argument(
    "-id",
    "--segID",
    required=True,
    type=str,
    help="string to identify the case",
  )
  parser.add_argument(
    "-wd",
    "--write_directory",
    type=str,
    default="",
    help="subdirectory to write to",
  )

  return parser.parse_args()

if __name__ == "__main__":

  args = inputs()
  if args.write_directory == "":
    args.write_directory = args.segID


  SURF_DIR = f"surfaces_lobes/{args.write_directory}"
  LABELMAP_DIR = f"images_lobes/{args.write_directory}"

  os.makedirs(SURF_DIR, exist_ok=True)
  os.makedirs(LABELMAP_DIR, exist_ok=True)

  device = torch.device("cpu")

  lobe_names = ["LUL", "LLL", "RUL", "RML", "RLL"]

  # Read image and load trained model
  input_img = utils.read_image(args.ct_path)
  model_hoff_lobes = mask.get_model("unet", "LTRCLobes")
  # segment image 
  labelmap = mask.apply(input_img, model_hoff_lobes)
  
  # check the index for all unique labels, as a debug check
  unique_labels = np.unique(labelmap)
  assert np.all(np.array([0, 1, 2, 3, 4, 5])==unique_labels), (
    f"unique labels not as expected {unique_labels}"
  )
  labelmap_img = sitk.GetImageFromArray(labelmap)
  labelmap_img.CopyInformation(input_img)

  out_dir = f"{LABELMAP_DIR}/unetsegmentation_{args.segID}_hofmanningerLTRClobes.mhd"
  sitk.WriteImage(
    labelmap_img,
    out_dir,
  )
  for l, label in enumerate([1, 2, 3, 4, 5]):
    print("-" * 30, "Writing label map", label)
    labelmap_arr_i = (labelmap == label).astype(int)
    labelmap_img_i = sitk.GetImageFromArray(labelmap_arr_i)
    labelmap_img_i = utils.fill_holes(labelmap_img_i, 6)
    labelmap_arr_i = sitk.GetArrayFromImage(labelmap_img_i)
    mesh = utils.numpy_to_surface(
      labelmap_arr_i,
      origin=input_img.GetOrigin(),
      spacing=input_img.GetSpacing(),
    )
    v.write(
      mesh,
      f"{SURF_DIR}/{args.segID}_mm_{lobe_names[l]}.stl",
    )

