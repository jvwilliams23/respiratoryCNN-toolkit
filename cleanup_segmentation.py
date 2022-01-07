"""
Use region-growing to extract upper airways from CNN segmentation
"""
from copy import copy
from glob import glob
from sys import exit

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vedo as v

from data import seg_half_dataset
from data import utils as u

"""
writeID is southamptonH04
bb 1 : (0, 41, 0, 512, 411, 963)
bb to lobes  : [36, 0, 67, 442, 353, 845]
sitk image shape (444, 355, 900)
np array shape (444, 355, 900), min -1024, max 3071
bounding box is [  0   0   0 444 355  90]
"""

MESH_FILE_NAME = "seg_mm_airway.vtk"
MESH_SMOOTHING_ITERATIONS = 1000
fig_dir = "debug/"


def copy_info_from_transposed_image(image, image_with_info_to_copy):
  """
  Take image which has its coordinates transposed but metadata in original
  alignment. Copy the information except size
  """
  image.SetSpacing(image_with_info_to_copy.GetSpacing())
  image.SetOrigin(image_with_info_to_copy.GetOrigin())
  return image


def binary_erode(img, kernel_radius=3):
  erode = sitk.BinaryErodeImageFilter()
  erode.SetKernelRadius(kernel_radius)  # -Set radius to that defined above
  erode.SetKernelType(2)  # -Set kernel shape to ball
  image_erode = erode.Execute(img)
  return image_erode


def binary_dilate(img, kernel_radius=3):
  dilate = sitk.BinaryDilateImageFilter()
  dilate.SetKernelRadius(kernel_radius)  # -Set radius to that defined above
  dilate.SetKernelType(2)  # -Set kernel shape to ball
  image_dilate = dilate.Execute(img)
  return image_dilate


def get_highest_point_in_z_dir(seg):
  seg_arr = sitk.GetArrayFromImage(seg).T
  # get all voxels containing airways
  label_1 = np.where(seg_arr == 1)
  # find location of airway voxel that is highest in z-dir (throat-region)
  highest_seed = label_1[2].max()
  # find where the highest seed is located in the array "label_1"
  highest_seed_loc_in_array = label_1[2].argmax()
  # get xyz voxel indexes of highest seed and use for region-growing
  seed = u.npToInts(np.array(label_1)[:, highest_seed_loc_in_array])
  seed_list = [seed]
  return seed_list


path = glob("images/H04-DICOM/*/")[0]
series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
  path, series_IDs[0]
)
series_reader = sitk.ImageSeriesReader()
series_reader.SetFileNames(series_file_names)
# Configure the reader to load all of the DICOM tags (public+private).
series_reader.MetaDataDictionaryArrayUpdateOn()
img = series_reader.Execute()  # -Get images

seg_orig = sitk.ReadImage("segmentations/seg-southamptonH04-airway.mhd")
# image axes are saved as z,y,x by accident, so we fix this here
seg_tmp_arr = sitk.GetArrayFromImage(seg_orig).T
seg_tmp = sitk.GetImageFromArray(seg_tmp_arr)
seg_tmp = copy_info_from_transposed_image(seg_tmp, seg_orig)
seg_orig = copy(seg_tmp)

seg = copy(seg_orig)
original_size = np.array(img.GetSize())
seg_original_size = np.array(seg.GetSize())
# get bounding-boxes used at each stage of cropping
bb_to_tissue = np.array([0, 41, 0, 512, 411, 963])
bb_to_lobes = np.array([36, 0, 67, 442, 353, 845])
output_img_shape = np.array([444, 355, 900])  # hard-coded for development

lhs_padding = bb_to_lobes[:3] + bb_to_tissue[:3]
rhs_padding = original_size - (bb_to_lobes[3:] + lhs_padding)
rhs_padding_orig = rhs_padding.copy()
rhs_padding[2] -= np.ceil(bb_to_lobes[-1] / 100) * 100 - bb_to_lobes[-1]
# rhs_padding = original_size - (output_img_shape + lhs_padding)
"""
During sliding box segmentation, we round the number of voxels in z-dir to 
the nearest 100 so we can easily create boxes of 10 voxels in height.
This was done by creating a padding on the upper bound. The code below crops
this extra padding in the z-direction, before we go on to pad all axes to 
account for cropping to lobes
"""
if rhs_padding[2] < 0:
  # initialise crop bounding box as xmin, ymin, zmin, xsize, ysize
  # min coordinates are 1 to account for padding of 1 in all directions,
  # and we subtract 2 from output img shape to account for an extra 1 layer pad
  bounding_box_to_crop = [1, 1, 1] + list(output_img_shape - 2)[:-1]
  # add zsize to cropping bounding box list
  bounding_box_to_crop.extend([bb_to_lobes[-1] + rhs_padding[2]])
  bounding_box_to_crop = u.npToInts(bounding_box_to_crop)
  seg = sitk.RegionOfInterest(
    seg,
    bounding_box_to_crop[int(len(bounding_box_to_crop) / 2) :],  # [::-1],
    bounding_box_to_crop[: int(len(bounding_box_to_crop) / 2)],  # [::-1],
  )
  # reset z-padding to account for cropping to lobes,
  # then subtract the padding left over from our sliding box pre-processing
  rhs_padding[2] = rhs_padding_orig[2] - rhs_padding[2]
  print("cropped size is ", seg.GetSize())
print("padding to upper bound is", rhs_padding)  # [::-1])

# pad segmentation to account for cropping to lobes
print("cropped size is ", seg.GetSize())
pad = sitk.ConstantPadImageFilter()
pad.SetPadLowerBound(u.npToInts(lhs_padding))  # [::-1]))
pad.SetPadUpperBound(u.npToInts(rhs_padding))  # [::-1]))
pad.SetConstant(0)
seg = pad.Execute(seg)
seg_repadded = copy(seg)
del pad
del seg_orig, seg_tmp, seg_tmp_arr
print("padded size is ", seg.GetSize())

assert img.GetSize() == seg.GetSize(), (
  "shape of img and segmentation do not match "
  f"{img.GetSize()} != {seg.GetSize()}"
)

print("getting largest island")
seg = binary_erode(seg)
seg = seg_half_dataset.getLargestIsland(seg)
seed_list = get_highest_point_in_z_dir(seg)
# sitk.WriteImage(seg, "seg.mhd")
print("island to arr")

"""
from myshow import *
img_arr = sitk.GetArrayFromImage(img)
print(seg_arr.shape, img_arr.shape)
slices_each_way = 10
for i, img_slice in enumerate(img_arr):
  if (i > seed[2] - slices_each_way) and (i < seed[2] + slices_each_way):
    img_slice_T = img_slice.T
    if i == seed[2]:
      print("seed at", seed)
      mywrite_slice_with_seed(img_slice_T, seed[:-1], f"{fig_dir}/debug_img_{i}.png")      
    else:
      mywrite_slice_subplots(img_slice_T, seg_arr[:,:,i], f"{fig_dir}/debug_img_{i}.png")
"""
# exit()

# pad = sitk.ConstantPadImageFilter()
# pad.SetPadLowerBound([1,1,1])
# pad.SetPadUpperBound([1,1,1])
# pad.SetConstant(200)
# img = pad.Execute(img)
region_growing_obj = sitk.ConnectedThreshold(
  img, seedList=seed_list, lower=-1200, upper=-950
)
del img
# clean up noise with dilation
print("performing dilate + erode to clean-up region-growing")
region_growing_obj = binary_dilate(region_growing_obj)
region_growing_obj = binary_erode(region_growing_obj)
region_growing_obj_arr = sitk.GetArrayFromImage(region_growing_obj)
assert region_growing_obj_arr.sum() != 0, "segmentation is empty"

# perform union of new upper airway segmentation and CNN segmentation
print("Calculating union of region-growing and CNN")
seg_repadded_arr = sitk.GetArrayFromImage(seg_repadded)
combined_vol_arr = seg_repadded_arr.copy()  # initialise union as CNN seg
combined_vol_arr += region_growing_obj_arr
combined_vol_arr = np.where(combined_vol_arr >= 1, 1, 0)
seed_list_new = get_highest_point_in_z_dir(region_growing_obj)
seed = seed_list_new[0]
assert (
  combined_vol_arr.T[seed[0], seed[1], seed[2]] == 1
), "seed does not correspond to an airway voxel"
del seg_repadded_arr, region_growing_obj_arr, region_growing_obj

combined_vol = sitk.GetImageFromArray(combined_vol_arr)
combined_vol.CopyInformation(seg_repadded)
del combined_vol_arr
# only keep voxels connected to upper airway seed (remove all noise from border)
region_growing_combined_vol = sitk.ConnectedThreshold(
  combined_vol, seedList=seed_list_new, lower=1.0, upper=1.0
)
region_growing_combined_vol_arr = sitk.GetArrayFromImage(
  region_growing_combined_vol
)
print("sum is", region_growing_combined_vol_arr.sum())
sitk.WriteImage(region_growing_combined_vol, "region-grow-union.mhd")
assert (region_growing_combined_vol_arr.sum() != 0) and (
  region_growing_combined_vol_arr.sum() != region_growing_combined_vol_arr.size
), "segmentation is empty"
del region_growing_combined_vol, combined_vol

print("converting arr to vol")
# write region-growing segmentation to surface file
vol = v.Volume(
  np.pad(region_growing_combined_vol_arr, 1), # pad to ensure closed surface
  spacing=seg_repadded.GetSpacing()
)
del region_growing_combined_vol_arr
print("vol to isosurface")
mesh = vol.isosurface()  # largest=True)
print("Writing vtk surface mesh")
v.write(mesh, MESH_FILE_NAME)

# perform smoothing on output surface file and over-write non-smooth version
surfs = pv.read(MESH_FILE_NAME)
print('smoothing')
surfs = surfs.smooth(n_iter=int(MESH_SMOOTHING_ITERATIONS))
print('saving', MESH_FILE_NAME)
surfs.save(MESH_FILE_NAME)

