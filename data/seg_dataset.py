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

def rg_based_crop_for_cnn(image, crop_fraction):
    """
    Use a connected-threshold estimator to separate background and foreground. 
    Then crop the image using the foreground's axis aligned bounding box.
    Parameters
    ----------
    image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    crop_fraction (float range 0:0.5) : fraction cropped from both sides in x-dir
    Return
    ------
    Cropped image based on foreground's axis aligned bounding box. 
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    lower, upper = -1500, -500 #Hard coded estimates
    # myshow(image>-200)

    image_island = getLargestIsland(image > -200) #> 50 & image < 2000 #uppe

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( image_island )
    bounding_box = label_shape_filter.GetBoundingBox(1) #-1 due to binary nature of threshold
    #-The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    roi = sitk.RegionOfInterest(image, 
                                bounding_box[int(len(bounding_box)/2):], 
                                bounding_box[0:int(len(bounding_box)/2)]
                                )
    # myshow(roi)
    size = np.array(roi.GetSize())
    seed = [int(size[0]//4), int(size[1]//2), int(size[2]//2)]
    rg = sitk.ConnectedThreshold(roi, seedList=[seed],
                              lower=-1000, upper=-500)


    # get largest internal island of air (should be lungs)
    image_island = getLargestIsland(rg) #> 50 & image < 2000 #uppe
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( image_island )

    # get bounding box as xmin, ymin, zmin, xmax, ymax, zmax
    # get bounding box as xstart, ystart, zstart, xsize, ysize, zsize
    bounding_box = list(label_shape_filter.GetBoundingBox(1)) #-1 due to binary nature of threshold
    print(bounding_box[0], bounding_box[3])
    bounding_x_size = bounding_box[3]
    bounding_box[0] = int(round(bounding_box[0] + bounding_x_size*crop_fraction))
    bounding_box[3] = int(round(bounding_x_size*(1-2*crop_fraction)))
    print(bounding_box[0], bounding_box[3])
    #-The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    # exit()
    roi = sitk.RegionOfInterest(roi, 
                                bounding_box[int(len(bounding_box)/2):], 
                                bounding_box[0:int(len(bounding_box)/2)]
                                )

    return roi

def getLargestIsland(segmentation):
  '''
  Take binary segmentation, as sitk.Image or np.ndarray,
  and return largest connected 'island'.
  '''
  seg_sitk = False
  if type(segmentation) == sitk.Image:
      seg_sitk = True
      segmentation = sitk.GetArrayFromImage(segmentation).astype(np.int16)
  
  labels = label(segmentation).astype(np.int16) #-get connected component
  assert( labels.max() != 0 ) # assume at least 1 connected component
  #-get largest connected region (converts from True/False to 1/0)
  largestIsland = np.array(
                          labels==np.argmax(np.bincount(labels.flat)[1:])+1,
                          dtype=np.int8
                          )
  #-if sitk.Image input, return type sitk.Image
  if seg_sitk:
      largestIsland = sitk.GetImageFromArray(largestIsland)
  return largestIsland


def resampleImage(imageIn, scanSize=[256, 256, 128], allAxes = True, interpolator=sitk.sitkLinear):
    '''
		Resamples image to improve quality and make isotropically spaced.
		Inputs:
				imageIn;         SimpleITK Image to be scaled, by a chosen factor.
				scanSize; 	     Desired output scan size.
				allAxes;         Boolean which determines whether only the width
				                 and height are scaled(False) or all 3 axes are 
								 respaced by the spacingFacotr (True).
	'''
    #-Euler transform to get extreme points to resample image
    euler3d = sitk.Euler3DTransform()
    euler3d.SetCenter(imageIn.TransformContinuousIndexToPhysicalPoint(np.array(imageIn.GetSize())/2.0))
    tx = 0
    ty = 0
    tz = 0
    euler3d.SetTranslation((tx, ty, tz))
    extreme_points = [
					imageIn.TransformIndexToPhysicalPoint((0,0,0)),
					imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),0,0)),
					imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),imageIn.GetHeight(),0)),
					imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),imageIn.GetHeight(),imageIn.GetDepth())),
					imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),0,imageIn.GetDepth())),
					imageIn.TransformIndexToPhysicalPoint((0,imageIn.GetHeight(),imageIn.GetDepth())),
					imageIn.TransformIndexToPhysicalPoint((0,0,imageIn.GetDepth())),
					imageIn.TransformIndexToPhysicalPoint((0,imageIn.GetHeight(),0))
					]
    inv_euler3d = euler3d.GetInverse()
    extreme_points_transformed = [inv_euler3d.TransformPoint(pnt) for pnt in extreme_points]

    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
    max_x = max(extreme_points_transformed)[0]
    max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    # print(output_spacing)
    if allAxes:
        output_spacing = (imageIn.GetSpacing()[0]*imageIn.GetSize()[0]/scanSize[0],
					imageIn.GetSpacing()[1]*imageIn.GetSize()[1]/scanSize[1],
					imageIn.GetSpacing()[2]*imageIn.GetSize()[2]/scanSize[2])
    else:
        output_spacing = (imageIn.GetSpacing()[0]*imageIn.GetSize()[0]/scanSize[0],
					imageIn.GetSpacing()[1]*imageIn.GetSize()[1]/scanSize[1],
					imageIn.GetSpacing()[2])
    # Identity cosine matrix (arbitrary decision).
    output_direction = imageIn.GetDirection()
    # Minimal x,y coordinates are the new origin.
    output_origin = imageIn.GetOrigin()
    # Compute grid size based on the physical size and spacing.
    output_size = [int((max_x-min_x)/output_spacing[0]), 
					int((max_y-min_y)/output_spacing[1]), 
					int((max_z-min_z)/output_spacing[2])]
    resampled_image = sitk.Resample(imageIn, output_size, euler3d, interpolator, 
									output_origin, output_spacing, output_direction)

    return resampled_image

class SegmentSet(data.Dataset):
  """
    list_scans is a list containing the filenames of scans
    scans_path and masks_path are the paths of the folders containing the data
    mode : 2d will return slices
  """
  def __init__(self, scans_path, crop_fraction=0.0):

    self.scans_path = scans_path
    self.crop_fraction = crop_fraction

  def __len__(self):
    return len(self.scans_path)

  def __getitem__(self):

    #load scan and mask
    ct_scan = utils.read_image(self.scans_path)
    ct_scanOrig = utils.read_image(self.scans_path)

    #ct_scan=sitk.GetImageFromArray(ct_scan)
    #ct_scan=resampleImage(ct_scanOrig, self.scan_size)
    #ct_scan=sitk.GetArrayFromImage(ct_scan)
    if self.crop_fraction != 0.:
      ct_scan = rg_based_crop_for_cnn(ct_scan, self.crop_fraction)
    #ct_scan=sitk.GetArrayFromImage(ct_scanOrig)
    minCutoff = -1000
    ct_scan=truncate(ct_scan, minCutoff, 600)
    ct_scan=(ct_scan-(minCutoff)) / 1600 # normalise HU

    return ct_scan[np.newaxis, :], ct_scanOrig


