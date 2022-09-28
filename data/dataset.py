import numpy as np
import nrrd, os, scipy.ndimage
from glob import glob
import torch
from torch.utils import data
import SimpleITK as sitk
import sys

from . import utils

def truncate(image, min_bound, max_bound):
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image

def resampleImage(imageIn, scanSize=[256, 256, 128], allAxes = True):
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
    resampled_image = sitk.Resample(imageIn, output_size, euler3d, sitk.sitkLinear, 
									output_origin, output_spacing, output_direction)

    return resampled_image

class Dataset(data.Dataset):
  """
    list_scans is a list containing the filenames of scans
    scans_path and masks_path are the paths of the folders containing the data
  """
  def __init__(self, scans_path, labels_path, scan_size = [128, 128, 128], n_classes = 5):
    self.scans_path = scans_path
    self.labels_path = labels_path
    self.scan_size = scan_size
    self.n_classes = n_classes

  def __len__(self):
    return len(self.scans_path)

  def __getitem__(self, index):

    #load scan and mask
    ct_scan, ct_orig, ct_space = utils.load_itk(self.scans_path[index])
    seg_mask, seg_orig, seg_space = utils.load_itk(self.labels_path[index])

    if self.n_classes == 5 or self.n_classes == 6:
      seg_mask[seg_mask == 0] = 0
      seg_mask[seg_mask < 4] = 0
      seg_mask[seg_mask == 4] = 1
      seg_mask[seg_mask == 5] = 2
      seg_mask[seg_mask == 6] = 3
      seg_mask[seg_mask == 7] = 4
      seg_mask[seg_mask == 8] = 5
      seg_mask[seg_mask > 8] = 0
    else:
      seg_mask[seg_mask <= 0] = 0
      seg_mask[seg_mask == 1] = 1
      seg_mask[seg_mask >  1] = 0
      
    minCutoff = -1000
    ct_scan=truncate(ct_scan, minCutoff, 600)
    ct_scan=(ct_scan-(minCutoff)) / 1600 # normalise HU

    return ct_scan[np.newaxis, :], seg_mask[np.newaxis, :]

