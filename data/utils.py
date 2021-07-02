import numpy as np
import torch
import SimpleITK as sitk
from copy import copy
from sys import exit
import vtk
from skimage.measure import label


# Not sure if works for all format (Tested only on mhd/zraw format)
def load_itk(filename):
  itkimage = sitk.ReadImage(filename)
  ct_scan = sitk.GetArrayFromImage(itkimage)
  origin = np.array(list(reversed(itkimage.GetOrigin())))
  spacing = np.array(list(reversed(itkimage.GetSpacing())))
  return ct_scan, origin, spacing

def lossTang2019(logits, labels, label, eps=1e-7):
  '''
    logits, labels, shape : [B, 1, Y, X]
  '''
  alpha = 1. #-params controlling weight for each class
  gamma = 5. #-params controlling weight for each class
  lam = 1. # controls balance between DICE and focal loss.
  #safe_tensor=torch.where(torch.isnan(logits),torch.zeros_like(logits),logits)
  #logits = safe_tensor
  #del safe_tensor
  logits=logits.clamp(eps,1.-eps)
  x = torch.where(labels==label, torch.ones(labels.shape), torch.zeros(labels.shape))

  numDice = torch.sum(logits * x)
  denomDice = torch.sum((logits *  x) + ((1 - logits)*x) + (logits*(1-x)))
  tmpDice = numDice/denomDice
  L_dice = 1-tmpDice #torch.sum(tmpDice)

  print("DICE IS", L_dice)

  numVox = logits.shape[-1]*logits.shape[-2]*logits.shape[-3]
  #print("x is",x)
  #print("logits is",logits)
  #print("numVox", numVox)
  #print(torch.log(logits))
  
  L_focal =  (-1*torch.sum(alpha * x * (1 - logits)**gamma * torch.log(logits)) / numVox )
  print("Focal is", L_focal)
  del x
  return (L_dice + lam * L_focal).type(torch.float16) #1 - torch.mean(num / (denom + eps))





def dice_loss_old(logits, labels, label, eps=1e-7):
  '''
    logits, labels, shape : [B, 1, Y, X]
  '''
  x = torch.where(labels==label, torch.ones(labels.shape), torch.zeros(labels.shape))
  
  num = 2. * torch.sum(logits * x)
  denom = torch.sum(logits**2. +  x**2.)
  return 1 - torch.mean(num / (denom + eps))

def dice(p, labels, label):
  x = torch.where(labels==label, torch.ones(labels.shape), torch.zeros(labels.shape))

  numDice = torch.sum(p * x)
  denomDice = torch.sum((p *  x) + ((1 - p)*x) + (p*(1-x)))
  tmpDice = numDice/denomDice
  L_dice = 1-tmpDice #torch.sum(tmpDice)

  del x
  return tmpDice


def dice_loss(output, labels, label):
  print("label is",label)
  with torch.no_grad():
    output[output!=label] = 0
    output[output==label] = 1
    labels[labels!=label] = 0
    labels[labels==label] = 1
  print("labels are", labels)
  num = 2. * torch.sum(output * labels)
  #print("intersecting", num)
  denom = torch.sum(output==1)+torch.sum(labels==1) #torch.sum(logits**2 + labels**2)
  #print("total volume", denom)  
  print("dice coeff", 1 - torch.true_divide(num, denom))
  return 1 - torch.true_divide(num, denom )


#def dice_loss(output, labels, label):
  
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
  labels = np.delete(labels, np.where(labels==0))
  dice = np.zeros(len(labels))
  for i, label in enumerate(labels):

    volume_sum = np.array(mask_gt==label).sum() + np.array(mask_pred==label).sum()
    if volume_sum == 0:
      dice[i] = 0
      continue
    volume_intersect = (np.array(mask_gt==label) & np.array(mask_pred==label)).sum()
    dice[i] = 2*volume_intersect / volume_sum

  return torch.mean(1 - torch.from_numpy( dice ) )

def getLargestIsland(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestIsland = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestIsland

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

        #-Gets image to be surfaced
        image = vtk.vtkImageThreshold()
        image.SetInputConnection(reader.GetOutputPort())
        image.Update()

        #-Creates surface from binary label map
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
