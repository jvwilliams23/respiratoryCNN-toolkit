"""
Split image in half with plane in X-direction. Segment each half with UNet then merge outputs

"""
import argparse
import logging
import numpy as np
from copy import copy
import hjson
import os
import torch
import SimpleITK as sitk
from data.seg_dataset import resampleImage

# from torch.utils import data
from glob import glob
from sys import exit
from os import mkdir
from distutils.util import strtobool
import vedo as v

if __name__ == "__main__":
  logging.basicConfig(
    filename="log_sliding_box_segmentation.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
  )
  logger = logging.getLogger(__name__)


from cleanup_tools import CleanupTools
import model
from data import *

def get_inputs():
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
    "-wd",
    "--write_dir",
    default="../segmentations/",
    type=str,
    help="output segmentation directory /path/to/segmentations/",
  )
  parser.add_argument(
    "-bd",
    "--bounding_box_dir",
    default="bounding_boxes/",
    type=str,
    help="Directory to save bounding boxes",
  )
  parser.add_argument(
    "-large",
    "--largest_only",
    default=False,
    action="store_true",
    help="Keep the largest connected component in the domain",
  )
  return parser.parse_args()


args = get_inputs()
with open(args.config_file) as f:
  config = hjson.load(f)
segID = config["segment3d"]["output_id"]

logging.getLogger("matplotlib.font_manager").disabled = True

logger.info(f"config {config}")
logger.info(f"argparser: {args}")
logger.info(f"data will be written to e.g.: {args.write_dir}/seg-{segID}-airway.mhd")

device = torch.device("cpu")

# mkdir("segmentations")
# mkdir(args.bounding_box_dir)

unet = model.UNet(
  1, config["train3d"]["n_classes"], config["train3d"]["start_filters"]
).to(device)
unet.load_state_dict(torch.load("./unet-model.pt"))

kwargs = {}
crop_to_lobes = bool(strtobool(config["segment3d"]["crop_to_lobes"][0]))
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
  # erode so that it is like cropping to airways
  kwargs["lobe_seg"] = CleanupTools.binary_erode(kwargs["lobe_seg"], 10)
  del lungs_arr, lobes_seg, lobes_arr

# for i in range(len(list_scans)):
downsampling_on = bool(strtobool(config["segment3d"]["downsampling_on"][0]))
if downsampling_on:
  downsampling_ratio = config["segment3d"]["downsample"]
  kwargs["downsampling_ratio"] = downsampling_ratio
kwargs["num_boxes"] = config["segment3d"]["num_boxes"]
kwargs["voxel_size"] = [0.5, 0.5, 0.5]

dataset = seg_half_dataset.SegmentSet(
  args.ct_path, downsample=downsampling_on, **kwargs
)
segID = config["segment3d"]["output_id"]
print("writeID is", segID)

(
  input_windows,
  origin_list,
  spacing,
  lower_list,
  mid_point_list,
  upper_list,
  bounding_box_to_tissue,
  bounding_box_to_lobes,
) = dataset.__getitem__()
# save bounding boxes to be used in cleanup
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
labelMap_windows = [None] * len(input_windows)
for w, window in enumerate(input_windows):
  window = torch.Tensor(
    np.array([window[np.newaxis, :].astype(np.float16)])
  ).to(
    device
  )  # scan
  logits = unet(window)
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
# so image axial direction is z-dir in sitk image
combined_vol = combined_vol.T
if args.largest_only:
  combined_vol = utils.extract_largest_island(combined_vol)
image_out = sitk.GetImageFromArray(combined_vol)
image_out.SetSpacing(spacing)
image_out.SetOrigin(origin_list[0])
print(f"output image size {image_out.GetSize()}")
print("Writing labelMap to mhd")
sitk.WriteImage(image_out, f"{args.write_dir}/seg-{segID}-airway.mhd", True)
print("numpy to volume")
mesh = utils.numpy_to_surface(
  sitk.GetArrayFromImage(image_out).T,
  spacing=image_out.GetSpacing(),
  origin=image_out.GetOrigin(),
  largest=args.largest_only,
)
print("Writing vtk surface mesh")
v.write(mesh, f"{args.write_dir}/{segID}_mm_airway.vtk")
