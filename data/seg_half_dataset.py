import logging
import numpy as np
import nrrd, os, scipy.ndimage
from glob import glob
import torch
from torch.utils import data
import SimpleITK as sitk
from math import floor, ceil
from copy import copy
from skimage import morphology
from skimage.measure import label, regionprops
from . import utils as u
import vedo as v

logger = logging.getLogger(__name__)

def truncate(image, min_bound, max_bound):
  image[image < min_bound] = min_bound
  image[image > max_bound] = max_bound
  return image

def visualise_bb(arr, spacing, origin):
  return v.Volume(arr, origin=origin, spacing=spacing).legosurface(
    vmin=0.5, vmax=1
  )

def numpy_to_ints(arr):
  return [int(arr[0]), int(arr[1]), int(arr[2])]

def is_integer(n):
  try:
    float(n)
  except ValueError:
    return False
  else:
    return float(n).is_integer()

def sliding_window_crop(image, num_boxes=5, crop_dir=2, overlap=2):
  """
  Use a connected-threshold estimator to separate background and foreground.
  Then crop the image using the foreground's axis aligned bounding box.
  Parameters
  ----------
  image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                               (the assumption underlying Otsu's method.)
  Return
  ------
  Cropped image based on foreground's axis aligned bounding box.
  """
  image_arr = sitk.GetArrayFromImage(image).T
  spacing = np.array(image.GetSpacing())  # [::-1]
  origin = np.array(image.GetOrigin())  # [::-1]
  lower_floor = False
  mid_floor = False
  upper_floor = False
  lower_list = []
  mid_point_list = []
  upper_list = []
  # image_island = getLargestIsland(image > -600)  # > 50 & image < 2000 #uppe

  # label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  # label_shape_filter.Execute(image_island)
  # bounding_box = np.array(label_shape_filter.GetBoundingBox(1))
  max_size = list(image.GetSize())  # [::-1]
  bounding_box = [0, 0, 0] + max_size
  bounding_box = np.array(bounding_box)
  roi_list = []
  # vp += v.Volume(sitk.GetArrayFromImage(image))
  # vp += visualise_bb(image_arr, spacing)
  # vp.show()
  # exit()
  print("sitk image shape", image.GetSize())

  bb_lower = bounding_box[0 : int(len(bounding_box) / 2)].T
  bb_upper = bounding_box[int(len(bounding_box) / 2) :].T
  bb_upper[crop_dir] = bb_upper[crop_dir] // num_boxes
  bb_mid = bb_upper // 2

  # roi_i = image_arr[bb_lower[2]:bb_upper[2], bb_lower[1]:bb_upper[1], bb_lower[0]:bb_upper[0]]
  roi_i = image_arr[
    bb_lower[0] : bb_upper[0],
    bb_lower[1] : bb_upper[1],
    bb_lower[2] : bb_upper[2],
  ]
  roi_list.append(roi_i)
  vol = copy(roi_i)
  # vp = v.Plotter()
  # vp += visualise_bb(vol, spacing, origin=origin)
  # check bounds of box are OK
  print(f"bounding box is {bounding_box}")
  print()
  print(f"lower : {bb_lower}. \t upper : {bb_upper}")
  # use alternating combination of ceil and floor to make spacing consistent
  if not is_integer(bb_lower[crop_dir]):
    if lower_floor:
      bb_lower[crop_dir] = floor(bb_lower[crop_dir])
      lower_floor = False
    else:
      bb_lower[crop_dir] = ceil(bb_lower[crop_dir])
      lower_floor = True
  if not is_integer(bb_mid[crop_dir]):
    if mid_floor:
      bb_mid[crop_dir] = floor(bb_mid[crop_dir])
      mid_floor = False
    else:
      bb_mid[crop_dir] = ceil(bb_mid[crop_dir])
      mid_floor = True
  if not is_integer(bb_upper[crop_dir]):
    if upper_floor:
      bb_upper[crop_dir] = floor(bb_upper[crop_dir])
      upper_floor = False
    else:
      bb_upper[crop_dir] = ceil(bb_upper[crop_dir])
      upper_floor = True
  box_size = bb_upper[crop_dir] - bb_lower[crop_dir]
  lower_list.append(bb_lower[crop_dir])
  mid_point_list.append(bb_mid[crop_dir])
  upper_list.append(bb_upper[crop_dir])

  origin_def = copy(origin)
  origin_list = []
  origin_list.append(copy(origin_def))

  # for box in range((num_boxes*2)-2):
  # for box in range(int(round(num_boxes*1.5))):
  for i in range((num_boxes * overlap) - (1 * overlap)):
    bb_lower[crop_dir] += box_size / overlap
    bb_mid[crop_dir] += box_size / overlap
    bb_upper[crop_dir] += box_size / overlap
    # use alternating combination of ceil and floor to make spacing consistent
    if not is_integer(bb_lower[crop_dir]):
      if lower_floor:
        bb_lower[crop_dir] = floor(bb_lower[crop_dir])
        lower_floor = False
      else:
        bb_lower[crop_dir] = ceil(bb_lower[crop_dir])
        lower_floor = True
    if not is_integer(bb_mid[crop_dir]):
      if mid_floor:
        bb_mid[crop_dir] = floor(bb_mid[crop_dir])
        mid_floor = False
      else:
        bb_mid[crop_dir] = ceil(bb_mid[crop_dir])
        mid_floor = True
    if not is_integer(bb_upper[crop_dir]):
      if upper_floor:
        bb_upper[crop_dir] = floor(bb_upper[crop_dir])
        upper_floor = False
      else:
        bb_upper[crop_dir] = ceil(bb_upper[crop_dir])
        upper_floor = True

    # prevent using a box too big by stopping loop if upper bound is too big
    if bb_upper[crop_dir] > max_size[crop_dir]:
      break
    # unsure why origin[crop_dir] gives weird results ...
    origin[crop_dir] = origin_def[crop_dir] + (
      bb_lower[crop_dir] * spacing[crop_dir]
    )
    origin_list.append(copy(origin))
    # get bounding box as xstart, ystart, zstart, xsize, ysize, zsize
    roi_i = image_arr[
      bb_lower[0] : bb_upper[0],
      bb_lower[1] : bb_upper[1],
      bb_lower[2] : bb_upper[2],
    ]

    # check bounds of box are OK
    print(
      f"lower : {bb_lower}. \t middle : {bb_mid} \t upper : {bb_upper}, size = {bb_upper[crop_dir]-bb_lower[crop_dir]}"
    )

    roi_list.append(roi_i)
    vol = copy(roi_i)
    lower_list.append(bb_lower[crop_dir])
    mid_point_list.append(bb_mid[crop_dir])
    upper_list.append(bb_upper[crop_dir])
    # vp += visualise_bb(vol, spacing, origin=origin)
    # break
  # vp.show()
  return roi_list, origin_list, lower_list, mid_point_list, upper_list


def rg_based_crop_for_cnn(image):
  """
  Use a connected-threshold estimator to separate background and foreground.
  Then crop the image using the foreground's axis aligned bounding box.
  Parameters
  ----------
  image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                               (the assumption underlying Otsu's method.)
  Return
  ------
  Cropped image based on foreground's axis aligned bounding box.
  """
  # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
  # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
  lower, upper = -1500, -500  # Hard coded estimates
  # myshow(image>-200)

  image_island = getLargestIsland(image > -200)  # > 50 & image < 2000 #uppe

  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute(image_island)
  bounding_box = label_shape_filter.GetBoundingBox(
    1
  )  # -1 due to binary nature of threshold
  # -The bounding box's first "dim" entries are the starting index and last "dim" entries the size
  roi = sitk.RegionOfInterest(
    image,
    bounding_box[int(len(bounding_box) / 2) :],
    bounding_box[0 : int(len(bounding_box) / 2)],
  )
  # myshow(roi)
  size = np.array(roi.GetSize())
  seed = [int(size[0] // 4), int(size[1] // 2), int(size[2] // 2)]
  rg = sitk.ConnectedThreshold(roi, seedList=[seed], lower=-1000, upper=-500)

  # get largest internal island of air (should be lungs) and
  # crop to lobes, so combined image will be size of two lungs
  image_island = getLargestIsland(rg)  # > 50 & image < 2000 #uppe
  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute(image_island)
  # get bounding box as xmin, ymin, zmin, xmax, ymax, zmax
  # get bounding box as xstart, ystart, zstart, xsize, ysize, zsize
  bounding_box_to_lobes = list(
    label_shape_filter.GetBoundingBox(1)
  )  # -1 due to binary nature of threshold

  # crop to lobes
  roi_to_lobes = sitk.RegionOfInterest(
    roi,
    bounding_box_to_lobes[int(len(bounding_box) / 2) :],
    bounding_box_to_lobes[0 : int(len(bounding_box) / 2)],
  )
  print("bb 1 :", bounding_box)
  print("bb to lobes  :", bounding_box_to_lobes)
  return roi_to_lobes, bounding_box, bounding_box_to_lobes

def rg_based_crop_to_pre_segmented_lobes(image, seg):
  """
  Use a pre-computed lobe segmentation to crop image before segmentation
  Parameters
  ----------
  image (SimpleITK image): A CT image
  seg (SimpleITK image): A binary labelmap of the lung hull
  Return
  ------
  Cropp
  """
  bounding_box_to_tissue = [0, 0, 0] + list(image.GetSize())
  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute(seg)
  bounding_box_to_lobes = list(label_shape_filter.GetBoundingBox(1))
  # -The bounding box's first "dim" entries are the starting index and last "dim" entries the size
  bounding_box_to_lobes[2] = 0
  bounding_box_to_lobes[-1] = image.GetSize()[-1]
  roi = sitk.RegionOfInterest(
    image,
    bounding_box_to_lobes[int(len(bounding_box_to_lobes) / 2) :],
    bounding_box_to_lobes[0 : int(len(bounding_box_to_lobes) / 2)],
  )
  print("bb 1 :", bounding_box_to_tissue)
  print("bb to lobes  :", bounding_box_to_lobes)
  print("roi size is", roi.GetSize())
  return roi, bounding_box_to_tissue, bounding_box_to_lobes


def getLargestIsland(segmentation):
  """
  Take binary segmentation, as sitk.Image or np.ndarray,
  and return largest connected 'island'.
  """
  if type(segmentation) == sitk.Image:
    seg_sitk = True
    labels = label(sitk.GetArrayFromImage(segmentation).astype(np.int16)).astype(np.int16)  # -get connected component
  else:
    labels = label(segmentation).astype(np.int16)  # -get connected component

  assert labels.max() != 0  # assume at least 1 connected component
  # -get largest connected region (converts from True/False to 1/0)
  largestIsland_arr = np.array(
    labels == np.argmax(np.bincount(labels.flat)[1:]) + 1, dtype=np.int8
  )
  del labels
  # -if sitk.Image input, return type sitk.Image
  if seg_sitk:
    largestIsland = sitk.GetImageFromArray(largestIsland_arr)
    del largestIsland_arr
    largestIsland.CopyInformation(segmentation)
    return largestIsland
  else:
    return largestIsland_arr


class SegmentSet(data.Dataset):
  """
  list_scans is a list containing the filenames of scans
  scans_path and masks_path are the paths of the folders containing the data
  """

  def __init__(self, scans_path, downsample=False, **kwargs):
    self.scans_path = scans_path
    self.downsample = downsample
    self.kwargs = kwargs

  def __len__(self):
    return len(self.scans_path)

  def __getitem__(self):

    # load scan and mask
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(self.scans_path)
    if series_IDs:  # -Sanity check
      series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        self.scans_path, series_IDs[0]
      )
      series_reader = sitk.ImageSeriesReader()
      series_reader.SetFileNames(series_file_names)
      # Configure the reader to load all of the DICOM tags (public+private).
      series_reader.MetaDataDictionaryArrayUpdateOn()
      ct_scanOrig = series_reader.Execute()  # -Get images
    else:
      ct_scanOrig = sitk.ReadImage(self.scans_path)
    sitk.ProcessObject_SetGlobalWarningDisplay(True)
    logger.info(f"image spacing: {ct_scanOrig.GetSpacing()}")
    logger.info(f"image origin: {ct_scanOrig.GetOrigin()}")
    logger.info(f"image size: {ct_scanOrig.GetSize()}")

    if "crop_to_lobes" in self.kwargs.keys():
        """
        (
          ct_scanOrig,
          bounding_box_to_tissue,
          bounding_box_to_lobes,
        ) = rg_based_crop_for_cnn(ct_scanOrig)
        """
        assert np.all(np.array(ct_scanOrig.GetSize()) == np.array(self.kwargs["lobe_seg"].GetSize())), (
          "shape of img and segmentation do not match "
          f"{ct_scanOrig.GetSize()} != {self.kwargs['lobe_seg'].GetSize()}"
        )
        (
          ct_scanOrig,
          bounding_box_to_tissue,
          bounding_box_to_lobes,
        ) = rg_based_crop_to_pre_segmented_lobes(ct_scanOrig, self.kwargs["lobe_seg"])
    else:
        bb_default = list(np.zeros(3)) + list(ct_scanOrig.GetSize())
        bounding_box_to_tissue = bb_default
        bounding_box_to_lobes = bb_default
    if self.downsample:
      ct_scanOrig = u.resample_image(ct_scanOrig, **self.kwargs)
      logger.info(f"downsampled image size is {ct_scanOrig.GetSize()}")

    # cropping works best if scan is multiple of 100, so we add some extra padding
    # in z-direction to facilitate this. Padding should always be > 1
    num_z_slices = ct_scanOrig.GetSize()[2]
    num_z_ceil_100 = int(ceil(num_z_slices / 100)) * 100
    num_z_to_pad = num_z_ceil_100 - num_z_slices - 1
    pad = sitk.ConstantPadImageFilter()
    pad.SetPadLowerBound((1, 1, 1))
    pad.SetPadUpperBound((1, 1, num_z_to_pad))
    pad.SetConstant(0)
    ct_scanOrig = pad.Execute(ct_scanOrig)
    # ct_scan=sitk.GetImageFromArray(ct_scan)
    #if self.downsample:
    #  ct_scanOrig = u.resample_image(ct_scanOrig, **self.kwargs)
    #  print(f"downsampled image size is {ct_scanOrig.GetSize()}")
    #   half = sitk.GetArrayFromImage(half)
    (
      roi_list,
      origin_list,
      lower_list,
      mid_point_list,
      upper_list,
    ) = sliding_window_crop(ct_scanOrig, self.kwargs["num_boxes"])
    for i, roi_i in enumerate(roi_list):
      minCutoff = -1000
      roi_i = truncate(roi_i, minCutoff, 600)
      logger.info(f"after truncation roi {i} min voxel {roi_list[i].min()}")
      logger.info(f"after truncation roi {i} max voxel {roi_list[i].max()}")
      roi_i = (roi_i - (minCutoff)) / 1600  # normalise HU
      roi_list[i] = roi_i
      logger.info(f"after scaling roi {i} min voxel {roi_list[i].min()}")
      logger.info(f"after scaling roi {i} max voxel {roi_list[i].max()}")

    return (
      roi_list,
      origin_list,
      ct_scanOrig.GetSpacing(),
      lower_list,
      mid_point_list,
      upper_list,
      bounding_box_to_tissue,
      bounding_box_to_lobes,
    )
