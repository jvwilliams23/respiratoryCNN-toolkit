import numpy as np
import nrrd, os, scipy.ndimage
from glob import glob
import torch
from torch.utils import data
import SimpleITK as sitk
from skimage import morphology
from skimage.measure import label, regionprops
from . import utils


def truncate(image, min_bound, max_bound):
  image[image < min_bound] = min_bound
  image[image > max_bound] = max_bound
  return image


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
  bounding_box = list(
    label_shape_filter.GetBoundingBox(1)
  )  # -1 due to binary nature of threshold

  # crop to lobes
  roi_to_lobes = sitk.RegionOfInterest(
    roi,
    bounding_box[int(len(bounding_box) / 2) :],
    bounding_box[0 : int(len(bounding_box) / 2)],
  )
  # get new bounding box to split in half
  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute(roi_to_lobes < -500)
  bounding_box = list(
    label_shape_filter.GetBoundingBox(1)
  )  # -1 due to binary nature of threshold

  # split into to halves for segmenting - - will reduce memory consumption
  print("bounding box sizes")
  print(bounding_box[0], bounding_box[3])
  bounding_x_size = bounding_box[3]
  bounding_box[3] = int(round(bounding_x_size * 0.5))
  print(bounding_box[0], bounding_box[3])
  roi_1 = sitk.RegionOfInterest(
    roi_to_lobes,
    bounding_box[int(len(bounding_box) / 2) :],
    bounding_box[0 : int(len(bounding_box) / 2)],
  )

  bounding_box[0] = int(round(bounding_x_size * 0.5))
  bounding_box[3] = int(round(bounding_x_size * 0.5))
  print(bounding_box[0], bounding_box[3])
  # -The bounding box's first "dim" entries are the starting index and last "dim" entries the size
  roi_2 = sitk.RegionOfInterest(
    roi_to_lobes,
    bounding_box[int(len(bounding_box) / 2) :],
    bounding_box[0 : int(len(bounding_box) / 2)],
  )
  return [roi_1, roi_2], roi_to_lobes


def getLargestIsland(segmentation):
  """
  Take binary segmentation, as sitk.Image or np.ndarray,
  and return largest connected 'island'.
  """
  if type(segmentation) == sitk.Image:
    seg_sitk = True
    segmentation = sitk.GetArrayFromImage(segmentation).astype(np.int16)

  labels = label(segmentation).astype(np.int16)  # -get connected component
  assert labels.max() != 0  # assume at least 1 connected component
  # -get largest connected region (converts from True/False to 1/0)
  largestIsland = np.array(
    labels == np.argmax(np.bincount(labels.flat)[1:]) + 1, dtype=np.int8
  )
  # -if sitk.Image input, return type sitk.Image
  if seg_sitk:
    largestIsland = sitk.GetImageFromArray(largestIsland)
  return largestIsland


def resampleImage(imageIn, interpolator=sitk.sitkLinear, **kwargs):
  """
  Resamples image to improve quality and make isotropically spaced.
  Inputs:
          SimpleITK Image; to be scaled, by a chosen factor.
          seedToChange;    a list of coordinates of a seed that may be influenced by any change in
                           slice index.
          voxel_size;         a float value to set new spacing to.
  """
  # -Euler transform to get extreme points to resample image
  euler3d = sitk.Euler3DTransform()
  euler3d.SetCenter(
    imageIn.TransformContinuousIndexToPhysicalPoint(
      np.array(imageIn.GetSize()) / 2.0
    )
  )
  tx = 0
  ty = 0
  tz = 0
  euler3d.SetTranslation((tx, ty, tz))
  extreme_points = [
    imageIn.TransformIndexToPhysicalPoint((0, 0, 0)),
    imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(), 0, 0)),
    imageIn.TransformIndexToPhysicalPoint(
      (imageIn.GetWidth(), imageIn.GetHeight(), 0)
    ),
    imageIn.TransformIndexToPhysicalPoint(
      (imageIn.GetWidth(), imageIn.GetHeight(), imageIn.GetDepth())
    ),
    imageIn.TransformIndexToPhysicalPoint(
      (imageIn.GetWidth(), 0, imageIn.GetDepth())
    ),
    imageIn.TransformIndexToPhysicalPoint(
      (0, imageIn.GetHeight(), imageIn.GetDepth())
    ),
    imageIn.TransformIndexToPhysicalPoint((0, 0, imageIn.GetDepth())),
    imageIn.TransformIndexToPhysicalPoint((0, imageIn.GetHeight(), 0)),
  ]
  inv_euler3d = euler3d.GetInverse()
  extreme_points_transformed = [
    inv_euler3d.TransformPoint(pnt) for pnt in extreme_points
  ]

  min_x = min(extreme_points_transformed)[0]
  min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
  min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
  max_x = max(extreme_points_transformed)[0]
  max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
  max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]
  # Use the original spacing (arbitrary decision).
  input_spacing = imageIn.GetSpacing()
  # print(output_spacing)
  if "downsampling_ratio" in kwargs.keys():
    output_spacing = np.array(input_spacing) * np.array(
      kwargs["downsampling_ratio"]
    )
    print(f"downsampling to {output_spacing}")
  elif "voxel_size" in kwargs.keys():
    voxel_size = kwargs["voxel_size"]
    if type(voxel_size) != float:
      output_spacing = voxel_size
    else:
      output_spacing = (voxel_size, voxel_size, voxel_size)
  # Identity cosine matrix (arbitrary decision).
  output_direction = imageIn.GetDirection()
  # Minimal x,y coordinates are the new origin.
  output_origin = imageIn.GetOrigin()
  # Compute grid size based on the physical size and spacing.
  output_size = [
    int((max_x - min_x) / output_spacing[0]),
    int((max_y - min_y) / output_spacing[1]),
    int((max_z - min_z) / output_spacing[2]),
  ]
  resampled_image = sitk.Resample(
    imageIn,
    output_size,
    euler3d,
    interpolator,
    output_origin,
    output_spacing,
    output_direction,
  )

  return resampled_image


class SegmentSet(data.Dataset):
  """
  list_scans is a list containing the filenames of scans
  scans_path and masks_path are the paths of the folders containing the data
  """

  def __init__(self, list_scans, scans_path, downsample=False, **kwargs):
    self.list_scans = list_scans
    self.scans_path = scans_path
    self.downsample = downsample
    self.kwargs = kwargs

  def __len__(self):
    return len(self.list_scans)

  def __getitem__(self, index):

    scan = self.list_scans[index]

    # load scan and mask
    path = self.scans_path
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    if series_IDs:  # -Sanity check
      print("READING DICOM")
      filetype = "DICOM"
      series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        path, series_IDs[0]
      )
      series_reader = sitk.ImageSeriesReader()
      series_reader.SetFileNames(series_file_names)
      # Configure the reader to load all of the DICOM tags (public+private).
      series_reader.MetaDataDictionaryArrayUpdateOn()
      ct_scanOrig = series_reader.Execute()  # -Get images
    else:
      ct_scanOrig = sitk.ReadImage(path + "/" + scan)

    # ct_scan=sitk.GetImageFromArray(ct_scan)
    if self.downsample:
      ct_scanOrig = resampleImage(ct_scanOrig, **self.kwargs)
      print(f"downsampled image size is {ct_scanOrig.GetSize()}")
    # ct_scan=sitk.GetArrayFromImage(ct_scan)
    ct_halfs, ct_scanOrig = rg_based_crop_for_cnn(ct_scanOrig)
    out_halfs = [None] * len(ct_halfs)
    for i, half in enumerate(ct_halfs):
      half = sitk.GetArrayFromImage(half)
      minCutoff = -1000
      half = truncate(half, minCutoff, 600)
      half = (half - (minCutoff)) / 1600  # normalise HU
      out_halfs[i] = half[np.newaxis, :]

    return out_halfs, ct_scanOrig