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
    mode : 2d will return slices
  """
  def __init__(self, list_scans, scans_path, masks_path, mode = "3d", scan_size = [128, 128, 128], n_classes = 5):
    self.list_scans = list_scans
    self.scans_path = scans_path
    self.masks_path = masks_path
    self.mode = mode
    self.scan_size = scan_size
    self.n_classes = n_classes

  def __len__(self):
    return len(self.list_scans)

  def __getitem__(self, index):

    scan = self.list_scans[index]

    #load scan and mask
    #path = os.path.join(self.scans_path, scan, '*', '*') ----- Original
    path = self.scans_path
    #scan_dicom_id = os.path.basename(glob(path)[0])   # used to find the corresponding lung mask ----- Original
    scan_dicom_id = scan[-8:-4]
    #nrrd_scan = nrrd.read(glob(os.path.join(path, "*CT.nrrd"))[0])   # tuple containing the CT scan and some metadata ----- Original
    #nrrd_scan = nrrd.read(glob(os.path.join(self.masks_path, scan_dicom_id + "*.nrrd"))[0])#-Load the LUNA16 ground truth scans. This worked, somehow.
    #seg_mask = np.swapaxes(nrrd_scan[0], 0, 2)
    print(str(self.masks_path)+str(scan_dicom_id)+ "*.nrrd")
    nrrd_scan = sitk.ReadImage(glob(os.path.join(self.masks_path, scan_dicom_id + "*.nrrd"))[0])
    #ct_scan = np.swapaxes(nrrd_scan[0], 0, 2) # JW 02/07/20
    if nrrd_scan.GetDepth()==0:
        print("read error: dataset.py")
        sys.exit(0)
    seg_space = nrrd_scan.GetSpacing()
    seg_orig = nrrd_scan.GetOrigin()
    seg_mask = sitk.GetArrayFromImage(nrrd_scan)
    #seg_mask = np.swapaxes(nrrd_scan[0], 0, 2)
    #seg_mask, _, _ = utils.load_itk(os.path.join(self.scans_path, scan_dicom_id + ".mhd"))# JW 02/07/19
    ct_scan, ct_orig, ct_space = utils.load_itk(glob(os.path.join(self.scans_path, "*" + scan_dicom_id + ".mhd"))[0])# function uses SimpleITK to load lung masks from mhd/zraw data


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
      
  
    if self.mode == "3d":
      ct_scan=sitk.GetImageFromArray(ct_scan)
      seg_mask=sitk.GetImageFromArray(seg_mask)
      #ct_scan=resampleImage(ct_scan, self.scan_size)
      #seg_mask=resampleImage(seg_mask, self.scan_size)
      ct_scan=sitk.GetArrayFromImage(ct_scan)
      seg_mask=sitk.GetArrayFromImage(seg_mask)
      if np.min(ct_scan.ravel()) < -1100:
        minCutoff = np.partition(np.unique(ct_scan.ravel()),2)[1]
      else:
        minCutoff=np.min(ct_scan.ravel())
      #minCutoff = -1000
      #ct_scan=truncate(ct_scan, minCutoff, -600)
      #ct_scan=(ct_scan-(minCutoff))/400 #abs(minCutoff) # normalise HU
      minCutoff = -1000
      ct_scan=truncate(ct_scan, minCutoff, 600)
      ct_scan=(ct_scan-(minCutoff)) / 1600 # normalise HU

    if self.mode == "2d":
      #return ct_scan[:, np.newaxis, :], seg_mask[:, np.newaxis, :]
      return ct_scan[np.newaxis, :], seg_mask[np.newaxis, :]
    else:
      return ct_scan[np.newaxis, :], seg_mask[np.newaxis, :]

