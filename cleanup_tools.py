from copy import copy
from glob import glob
from sys import exit

import numpy as np
import pyvista as pv
import SimpleITK as sitk
from skimage import morphology, transform, filters
from skimage.measure import label, regionprops
from scipy.spatial.transform import Rotation as R
import skan
from scipy.ndimage import rotate
import scipy.sparse.csgraph as scigraph
import networkx as nx
import networkx.algorithms.approximation.clique as clique
import vedo as v

from data import seg_half_dataset
from data import utils as u
import userUtils as utils


class CleanupTools:
  def __init__(self, ct_original, segmentation, segID, z_ind=0, **kwargs):
    self.ct_original = ct_original
    self.segmentation = segmentation
    self.segID = segID
    self.kwargs = kwargs
    self._ORIGINAL_SIZE = np.array(ct_original.GetSize())

  def cleanup_cropped_segmentation(self, ct_original, segmentation, **kwargs):
    seg_padded = self._repad_cropped_segmentation(segmentation)  # **kwargs)
    seg_cropped, bounding_box = self._crop_edges(seg_padded)
    seg_cropped = seg_half_dataset.getLargestIsland(seg_cropped)
    # del seg_padded
    seg_repadded = self._repad_cropped_segmentation_post(
      seg_cropped, bounding_box
    )
    region_grown_seg = self.regiongrowing_upper_airways(
      ct_original, seg_repadded
    )
    combined_vol = self.union_of_unet_and_region_growing(
      seg_repadded, region_grown_seg
    )
    return combined_vol, seg_repadded, region_grown_seg

  def regiongrowing_upper_airways(self, image, segmentation):
    assert image.GetSize() == segmentation.GetSize(), (
      "shape of img and segmentation do not match "
      f"{image.GetSize()} != {segmentation.GetSize()}"
    )
    seed = CleanupTools.get_highest_point_in_z_dir(segmentation)
    region_growing_obj = sitk.ConnectedThreshold(
      image, seedList=[seed], lower=-1200, upper=-950
    )
    region_growing_obj = CleanupTools.binary_dilate(region_growing_obj)
    region_growing_obj = CleanupTools.binary_erode(region_growing_obj)
    region_growing_obj_arr = sitk.GetArrayFromImage(region_growing_obj)
    assert region_growing_obj_arr.sum() != 0, "segmentation is empty"
    del region_growing_obj_arr
    return region_growing_obj

  def union_of_unet_and_region_growing(self, unet_seg, rg_seg):
    seg_repadded_arr = sitk.GetArrayFromImage(unet_seg)
    combined_vol_arr = seg_repadded_arr.copy()  # initialise union as CNN seg
    region_growing_obj_arr = sitk.GetArrayFromImage(rg_seg)
    combined_vol_arr += region_growing_obj_arr
    combined_vol_arr = np.where(combined_vol_arr >= 1, 1, 0)
    seed = CleanupTools.get_highest_point_in_z_dir(rg_seg)
    assert (
      combined_vol_arr.T[seed[0], seed[1], seed[2]] == 1
    ), "seed does not correspond to an airway voxel"
    del seg_repadded_arr

    combined_vol = sitk.GetImageFromArray(combined_vol_arr)
    combined_vol.CopyInformation(unet_seg)
    del combined_vol_arr
    # apply small dilation as the mesh will be eroded by surface smoothing
    # combined_vol = binary_dilate(combined_vol, kernel_radius=1)
    # only keep voxels connected to upper airway seed (remove all noise from border)
    region_growing_combined_vol = sitk.ConnectedThreshold(
      combined_vol, seedList=[seed], lower=1.0, upper=1.0
    )
    return region_growing_combined_vol

  def _repad_cropped_segmentation(self, segmentation, **kwargs):
    seg_original_size = np.array(segmentation.GetSize())
    # get bounding-boxes used at each stage of cropping
    """ TODO need more pythonic way to get bounding boxes """
    bb_to_tissue = np.loadtxt(f"bounding_box_to_tissue-{self.segID}.txt")
    bb_to_lobes = np.loadtxt(f"bounding_box_to_lobes-{self.segID}.txt")
    output_img_shape = np.array(segmentation.GetSize())

    lhs_padding = bb_to_lobes[:3] + bb_to_tissue[:3]
    rhs_padding = self._ORIGINAL_SIZE - (bb_to_lobes[3:] + lhs_padding)
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
      segmentation = sitk.RegionOfInterest(
        segmentation,
        bounding_box_to_crop[int(len(bounding_box_to_crop) / 2) :],
        bounding_box_to_crop[: int(len(bounding_box_to_crop) / 2)],
      )
      # reset z-padding to account for cropping to lobes,
      # then subtract the padding left over from our sliding box pre-processing
      rhs_padding[2] = rhs_padding_orig[2] - rhs_padding[2]
      print("cropped size is ", segmentation.GetSize())
    print("padding to upper bound is", rhs_padding)

    # pad segmentation to account for cropping to lobes
    print("cropped size is ", segmentation.GetSize())
    pad = sitk.ConstantPadImageFilter()
    pad.SetPadLowerBound(u.npToInts(lhs_padding))
    pad.SetPadUpperBound(u.npToInts(rhs_padding))
    pad.SetConstant(0)
    segmentation = pad.Execute(segmentation)
    seg_repadded = copy(segmentation)
    return seg_repadded

  def _repad_cropped_segmentation_post(
    self, segmentation, bounding_box, **kwargs
  ):
    """
    For segmentation cropped in post-processing.
    Assumes cropping is symmetrical in upper/lower sides
    """
    seg_original_size = np.array(segmentation.GetSize())

    lhs_padding = bounding_box[:3]
    rhs_padding = copy(lhs_padding)  # symmetrical in lower-upper dirs
    if np.any(
      seg_original_size + lhs_padding + rhs_padding != self._ORIGINAL_SIZE
    ):
      diff = self._ORIGINAL_SIZE - (
        seg_original_size + lhs_padding + rhs_padding
      )
      print("repadding not equal!")
      print("difference is", diff)
      # print((seg_original_size+lhs_padding+rhs_padding+diff))
      # print((seg_original_size+lhs_padding+rhs_padding-diff))
      # for now, we just make up the difference by adding to upper side
      rhs_padding += diff

    # pad segmentation to account for cropping to lobes
    print("cropped size is ", segmentation.GetSize())
    print(f"bounding_box is {bounding_box}")
    pad = sitk.ConstantPadImageFilter()
    pad.SetPadLowerBound(u.npToInts(lhs_padding))
    pad.SetPadUpperBound(u.npToInts(rhs_padding))
    pad.SetConstant(0)
    segmentation = pad.Execute(segmentation)
    seg_repadded = copy(segmentation)
    return seg_repadded

  def _crop_edges(self, segmentation):
    size = np.array(segmentation.GetSize())
    # pct_to_crop_each_side = 10.0
    pct_to_crop_each_side = np.array([10, 30, 10])
    size_each_side = size / 2
    lower_bound = np.round(np.zeros(3) + size * pct_to_crop_each_side // 100.0)
    upper_bound = np.round(size - 2.0 * size * pct_to_crop_each_side // 100.0)

    bounding_box = list(u.npToInts(lower_bound)) + list(u.npToInts(upper_bound))

    roi = sitk.RegionOfInterest(
      segmentation,
      bounding_box[int(len(bounding_box) / 2) :],
      bounding_box[0 : int(len(bounding_box) / 2)],
    )
    print("cropped size is", roi.GetSize())
    return roi, bounding_box

    # seg = copy(roi)
    # seg = binary_erode(seg_tmp)
    # seg = seg_half_dataset.getLargestIsland(seg)

  # @staticmethod
  # def get_highest_point_in_z_dir(segmentation):
  #   seg_arr = sitk.GetArrayFromImage(segmentation).T
  #   # get all voxels containing airways
  #   label_1 = np.where(seg_arr == 1)
  #   # find location of airway voxel that is highest in z-dir (throat-region)
  #   highest_seed = label_1[2].max()
  #   # find where the highest seed is located in the array "label_1"
  #   highest_seed_loc_in_array = label_1[2].argmax()
  #   # get xyz voxel indexes of highest seed and use for region-growing
  #   seed = u.npToInts(np.array(label_1)[:, highest_seed_loc_in_array])
  #   seed_list = [seed]
  #   return seed_list

  @staticmethod
  def image_to_mesh(segmentation):
    seg_arr = sitk.GetArrayFromImage(segmentation)
    vol = v.Volume(
      np.pad(seg_arr, 1),  # pad to ensure closed surface
      spacing=segmentation.GetSpacing(),
    )
    print("vol to isosurface")
    mesh = vol.isosurface()  # largest=True)
    del vol
    return mesh

  @staticmethod
  def copy_info_from_transposed_image(image, image_with_info_to_copy):
    """
    Take image which has its coordinates transposed but metadata in original
    alignment. Copy the information except size
    """
    image.SetSpacing(image_with_info_to_copy.GetSpacing())
    image.SetOrigin(image_with_info_to_copy.GetOrigin())
    return image

  @staticmethod
  def get_highest_point_in_z_dir(segmentation):
    seg_arr = sitk.GetArrayFromImage(segmentation).T
    # get all voxels containing airways
    label_1 = np.where(seg_arr == 1)
    # find location of airway voxel that is highest in z-dir (throat-region)
    highest_seed = label_1[2].max()
    # find where the highest seed is located in the array "label_1"
    highest_seed_loc_in_array = label_1[2].argmax()
    # get xyz voxel indexes of highest seed and use for region-growing
    seed = u.npToInts(np.array(label_1)[:, highest_seed_loc_in_array])
    return seed

  @staticmethod
  def binary_erode(img, kernel_radius=3):
    erode = sitk.BinaryErodeImageFilter()
    erode.SetKernelRadius(kernel_radius)  # -Set radius to that defined above
    erode.SetKernelType(2)  # -Set kernel shape to ball
    image_erode = erode.Execute(img)
    return image_erode

  @staticmethod
  def binary_dilate(img, kernel_radius=3):
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(kernel_radius)  # -Set radius to that defined above
    dilate.SetKernelType(2)  # -Set kernel shape to ball
    image_dilate = dilate.Execute(img)
    return image_dilate


if __name__ == "__main__":
  z_ind = 0
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

  # seg_orig = sitk.ReadImage("../seg_data/seg-southamptonA01-nocrop-airway.mhd")
  # image axes are saved as z,y,x by accident, so we fix this here
  seg_new_arr = sitk.GetArrayFromImage(seg_orig).T
  seg_new = sitk.GetImageFromArray(seg_new_arr)
  seg_new = CleanupTools.copy_info_from_transposed_image(seg_new, seg_orig)

  print("original")
  print(seg_orig.GetSize(), seg_orig.GetSpacing())

  print("new")
  print(seg_new.GetSize(), seg_new.GetSpacing())

  ctools = CleanupTools(img, seg_new, "southamptonH04")
  (
    combined_vol,
    seg_repadded,
    region_grown_seg,
  ) = ctools.cleanup_cropped_segmentation(img, seg_new)
  print(f"spacing region_grown {region_grown_seg.GetSpacing()}")
  print(f"spacing seg_repadded {seg_repadded.GetSpacing()}")
  print(f"spacing combined_vol {combined_vol.GetSpacing()}")
  del ctools
  mesh = CleanupTools.image_to_mesh(combined_vol)
  v.write(mesh, "newclean-combinedH04.vtk")
  del mesh, combined_vol
  mesh = CleanupTools.image_to_mesh(seg_repadded)
  v.write(mesh, "newclean-unetH04.vtk")
  del mesh, seg_repadded
  mesh = CleanupTools.image_to_mesh(region_grown_seg)
  v.write(mesh, "newclean-regiongrowH04.vtk")
  del mesh, region_grown_seg
