import numpy as np
import nrrd, os, scipy.ndimage
from glob import glob
import torch
from torch.utils import data
import SimpleITK as sitk
from . import utils

def truncate(image, min_bound, max_bound):
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image

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
  def __init__(self, list_scans, scans_path, mode = "3d", scan_size=[128, 128, 128]):
    self.list_scans = list_scans
    self.scans_path = scans_path
    self.mode = mode
    self.scan_size = scan_size

  def __len__(self):
    return len(self.list_scans)

  def __getitem__(self, index):

    scan = self.list_scans[index]

    #load scan and mask
    path = self.scans_path
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    if series_IDs: #-Sanity check
      print("READING DICOM")
      filetype = "DICOM"
      series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_IDs[0])
      series_reader = sitk.ImageSeriesReader()
      series_reader.SetFileNames(series_file_names)
      # Configure the reader to load all of the DICOM tags (public+private).
      series_reader.MetaDataDictionaryArrayUpdateOn()
      ct_scanOrig = series_reader.Execute() #-Get images
    else:
      ct_scanOrig = sitk.ReadImage(path+"/"+scan)

    #ct_scan=sitk.GetImageFromArray(ct_scan)
    ct_scan=resampleImage(ct_scanOrig, self.scan_size)
    ct_scan=sitk.GetArrayFromImage(ct_scan)
    minCutoff = -1000
    ct_scan=truncate(ct_scan, minCutoff, 600)
    ct_scan=(ct_scan-(minCutoff)) / 1600 # normalise HU

    return ct_scan[np.newaxis, :], ct_scanOrig


