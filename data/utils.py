import numpy as np
import torch
import SimpleITK as sitk
from copy import copy
from sys import exit
import vtk
import vedo as v
from skimage.measure import label

def npToInts(arr):
  return [int(a) for a in arr]

# Not sure if works for all format (Tested only on mhd/zraw format)
def load_itk(filename):
  itkimage = sitk.ReadImage(filename)
  ct_scan = sitk.GetArrayFromImage(itkimage)
  origin = np.array(list(reversed(itkimage.GetOrigin())))
  spacing = np.array(list(reversed(itkimage.GetSpacing())))
  return ct_scan, origin, spacing

def rg_based_crop(image):
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

  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute(image)
  bounding_box = label_shape_filter.GetBoundingBox(
    1
  )  # -1 due to binary nature of threshold
  # -The bounding box's first "dim" entries are the starting index and last "dim" entries the size
  roi = sitk.RegionOfInterest(
    image,
    bounding_box[int(len(bounding_box) / 2) :],
    bounding_box[0 : int(len(bounding_box) / 2)],
  )
  return roi

def lossTang2019(logits, labels, label, eps=1e-7, gamma=5.0):
  """
  logits, labels, shape : [B, 1, Y, X]
  gamma (float): controls focusing on training under-represented voxels ('hard' to learn)
  """
  alpha = 1.0  # -params controlling weight for each class
  lam = 1.0  # controls balance between DICE and focal loss.
  # safe_tensor=torch.where(torch.isnan(logits),torch.zeros_like(logits),logits)
  # logits = safe_tensor
  # del safe_tensor
  logits = logits.clamp(eps, 1.0 - eps)
  x = torch.where(
    labels == label, torch.ones(labels.shape), torch.zeros(labels.shape)
  )

  numDice = torch.sum(logits * x)
  denomDice = torch.sum((logits * x) + ((1 - logits) * x) + (logits * (1 - x)))
  tmpDice = numDice / denomDice
  L_dice = 1 - tmpDice  # torch.sum(tmpDice)

  print("DICE IS", L_dice)

  numVox = logits.shape[-1] * logits.shape[-2] * logits.shape[-3]

  L_focal = (
    -1
    * torch.sum(alpha * x * (1 - logits) ** gamma * torch.log(logits))
    / numVox
  )
  print("Focal is", L_focal)
  del x
  return (L_dice + lam * L_focal).type(
    torch.float16
  )  # 1 - torch.mean(num / (denom + eps))


def dice_loss_old(logits, labels, label, eps=1e-7):
  """
  logits, labels, shape : [B, 1, Y, X]
  """
  x = torch.where(
    labels == label, torch.ones(labels.shape), torch.zeros(labels.shape)
  )

  num = 2.0 * torch.sum(logits * x)
  denom = torch.sum(logits ** 2.0 + x ** 2.0)
  return 1 - torch.mean(num / (denom + eps))


def dice(p, labels, label):
  x = torch.where(
    labels == label, torch.ones(labels.shape), torch.zeros(labels.shape)
  )

  numDice = torch.sum(p * x)
  denomDice = torch.sum((p * x) + ((1 - p) * x) + (p * (1 - x)))
  tmpDice = numDice / denomDice
  L_dice = 1 - tmpDice  # torch.sum(tmpDice)

  del x
  return tmpDice


def dice_loss(output, labels, label):
  print("label is", label)
  with torch.no_grad():
    output[output != label] = 0
    output[output == label] = 1
    labels[labels != label] = 0
    labels[labels == label] = 1
  print("labels are", labels)
  num = 2.0 * torch.sum(output * labels)
  # print("intersecting", num)
  denom = torch.sum(output == 1) + torch.sum(
    labels == 1
  )  # torch.sum(logits**2 + labels**2)
  # print("total volume", denom)
  print("dice coeff", 1 - torch.true_divide(num, denom))
  return 1 - torch.true_divide(num, denom)


# def dice_loss(output, labels, label):

#    logits, labels, shape : [B, 1, Y, X]
#
#
#  num = 2. * torch.sum((output==label)==(labels==label))
#  print("intersecting", num)
#  denom = torch.sum(output==label)+torch.sum(labels==label) #torch.sum(logits**2 + labels**2)
#  print("total volume", denom)
#  print("dice coeff", 1 - torch.true_divide(num, denom))
#  return 1 - torch.true_divide(num, denom )


def compute_dice_coefficient_np(mask_pred, mask_gt):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  mask_gt = np.array(mask_gt)
  mask_pred = np.array(mask_pred)
  labels = np.unique(mask_gt)
  labels = np.delete(labels, np.where(labels == 0))
  dice = np.zeros(len(labels))
  for i, label in enumerate(labels):

    volume_sum = (
      np.array(mask_gt == label).sum() + np.array(mask_pred == label).sum()
    )
    if volume_sum == 0:
      dice[i] = 0
      continue
    volume_intersect = (
      np.array(mask_gt == label) & np.array(mask_pred == label)
    ).sum()
    dice[i] = 2 * volume_intersect / volume_sum

  return torch.mean(1 - torch.from_numpy(dice))


def getLargestIsland(segmentation):
  labels = label(segmentation)
  assert labels.max() != 0  # assume at least 1 CC
  largestIsland = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
  del labels
  return largestIsland


def numpy_to_surface(arr, spacing=[1, 1, 1], origin=[0, 0, 0]):
  vol = v.Volume(arr, spacing=spacing, origin=origin)
  return vol.isosurface(largest=True)

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

def binaryLabelToSTL(image, outputName="newairway.stl"):
  """
  .STL file creation from a .mhd file containing a binary label map.
  Modification of the above code.
  """

  filename = image

  # Read the .mhd image with vtk
  reader = vtk.vtkMetaImageReader()
  reader.SetFileName(filename)
  reader.Update()

  # -Gets image to be surfaced
  image = vtk.vtkImageThreshold()
  image.SetInputConnection(reader.GetOutputPort())
  image.Update()

  # -Creates surface from binary label map
  surface = vtk.vtkDiscreteMarchingCubes()
  surface.SetInputConnection(image.GetOutputPort())
  surface.GenerateValues(1, 1, 1)
  surface.Update()

  writer = vtk.vtkSTLWriter()
  writer.SetInputConnection(surface.GetOutputPort())
  # writer.SetFileTypeToBinary()
  writer.SetFileName(outputName)
  writer.Write()
  return None
