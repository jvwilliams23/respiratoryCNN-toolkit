import numpy as np
import torch
import SimpleITK as sitk
from copy import copy
from sys import exit
import vtk
import vedo as v
from skimage.measure import label

import numpy as np
import torch

def convert_text_to_filenames(file_string):
  """
  read a tabular txt file formatted as follows:
    /path/to/image1 /path/to/label1
    /path/to/image2 /path/to/label2
 
  Output
  ------
  image_names (list of str): all strings with path to image
  label_names (list of str): all strings with path to label
  """
  with open(file_string, "r") as f:
    all_cases = f.readlines()

  image_file_list = []
  label_file_list = []
  for case_row in all_cases:
    removed_newline = case_row.replace("\n", "")
    # print(removed_newline)
    # check if string is empty (meaning no line in file)
    if removed_newline.__eq__(""):
      continue
    image_file_list.append(removed_newline.split()[0])
    label_file_list.append(removed_newline.split()[1])
  return image_file_list, label_file_list


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def np_to_ints(arr):
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

  #print(f"\tClass {label} dice loss is {L_dice.detach().numpy()}")
  #print(f"\tClass {label} dice loss is {L_dice}")
  print(f"\tClass {label} dice loss is", L_dice)

  numVox = logits.shape[-1] * logits.shape[-2] * logits.shape[-3]

  L_focal = (
    -1
    * torch.sum(alpha * x * (1 - logits) ** gamma * torch.log(logits))
    / numVox
  )
  #print(f"\tClass {label} focal loss is {L_focal.detach().numpy()}")
  #print(f"\tClass {label} focal loss is {L_focal}")
  print(f"\tClass {label} focal loss is", L_focal)

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

def extract_largest_island(segmentation):
  """Take binary segmentation, as sitk.Image or np.ndarray, and return largest
  connected 'island'."""
  seg_sitk = False
  if type(segmentation) == sitk.Image:
    seg_sitk = True
    segOrig = copy(segmentation)
    # logger.info("extract_largest_island, changing sitk.Image to array")
    segmentation = sitk.GetArrayFromImage(segmentation).astype(np.int16)

  tot_voxel_num = segmentation.size
  labels = label(segmentation)#.astype(np.int8)  # -get connected component
  assert labels.max() != 0  # assume at least 1 connected component
  # -get largest connected region (converts from True/False to 1/0)
  unique_labels = np.array(np.unique(labels.flat, return_counts=True))
  #print("unique labels are")
  #print(unique_labels)
  #max_label = unique_labels[0][unique_labels[1].argmax()]
  max_label = unique_labels[0][np.argsort(unique_labels[1], axis=0)[-2]]
  #max_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
  #print("max label is ", max_label)
  largestIsland = np.array(
    labels == max_label, dtype=np.int8
  )
  largestIsland_vox_num = largestIsland.sum()
  #print("island to total volume ratio", largestIsland_vox_num / tot_voxel_num)
  # -if sitk.Image input, return type sitk.Image
  if seg_sitk:
    largestIsland = sitk.GetImageFromArray(largestIsland)
    largestIsland.CopyInformation(segOrig)
  return largestIsland

def numpy_to_surface(arr, spacing=[1, 1, 1], origin=[0, 0, 0], largest=True):
  vol = v.Volume(arr, spacing=spacing, origin=origin)
  return vol.isosurface(largest=largest)

def resample_image(imageIn, interpolator=sitk.sitkLinear, **kwargs):
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

  # Use the original spacing (arbitrary decision).
  input_spacing = imageIn.GetSpacing()
  input_size = imageIn.GetSize()
  # print(output_spacing)
  if "downsampling_ratio" in kwargs.keys():
    output_spacing = np.array(input_spacing) * np.array(
      kwargs["downsampling_ratio"]
    )
    print(f"downsampling to {output_spacing}")
  if "voxel_size" in kwargs.keys():
    voxel_size = kwargs["voxel_size"]
    print("voxel size", voxel_size)
    print("type voxel size", type(voxel_size))
    # check that voxel size is list, tuple or np array 
    try:
      len(voxel_size)
    except TypeError:
      raise TypeError(
        f"resample_image kwargs 'voxel_size' type {type(voxel_size)} be list\n"
        f"value is currently {voxel_size}"
      )
    output_spacing = voxel_size
  print("out spacing is", output_spacing)
  # Identity cosine matrix (arbitrary decision).
  output_direction = imageIn.GetDirection()
  # Minimal x,y coordinates are the new origin.
  output_origin = imageIn.GetOrigin()
  # Compute grid size based on the physical size and spacing.
  output_size = np_to_ints([
    input_size[0] * input_spacing[0] / output_spacing[0],
    input_size[1] * input_spacing[1] / output_spacing[1],
    input_size[2] * input_spacing[2] / output_spacing[2],
  ])
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
