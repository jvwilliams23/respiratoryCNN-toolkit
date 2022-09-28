import numpy as np
import hjson as json
import os
import torch
import SimpleITK as sitk
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from glob import glob
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sys import exit

from enet import ENet
from model import UNet

from data import *

with open("trainconfig.json") as f:
  config = json.load(f)

device = torch.device("cpu")

if os.path.isfile("train_cases.txt") and os.path.isfile("val_cases.txt"):
  train_scans, train_labels = utils.convert_text_to_filenames("train_cases.txt")
  val_scans, val_labels = utils.convert_text_to_filenames("val_cases.txt")
else:
  image_file_list, label_file_list = utils.convert_text_to_filenames(
    "cnn_training_cases.txt"
  )
  train_scans, val_scans, train_labels, val_labels = train_test_split(
    image_file_list,
    label_file_list,
    test_size=config["train3d"]["validation_size"],
    shuffle=True,
  )

# pack all dataset info into one list for readability
dataset_params = [
  config["train3d"]["scan_size"],
  config["train3d"]["n_classes"],
]
# read data splits
train_data = dataset.Dataset(train_scans, train_labels, *dataset_params)
print("train_data OK. Number of scans: ", len(train_data))
val_data = dataset.Dataset(val_scans, val_labels, *dataset_params)

# select which model to train
if config["train3d"]["model"].lower() == "unet":
  print("Testing UNet")
  model = UNet(
    1,
    config["train3d"]["n_classes"],
    config["train3d"]["start_filters"],
    bilinear=False,
  ).to(device)
  model_string = "unet-"  # string to use for saving the model
elif config["train3d"]["model"].lower() == "enet":
  print("Testing ENet")
  model = ENet(config["train3d"]["n_classes"]).to(device)
  model_string = "enet"  # string to use for saving the model
else:
  raise AssertionError("Unrecognised model. Exiting.")

model.load_state_dict(torch.load(f"./{model_string}model.pt"))

for i in range(len(val_data)):
  print(f"validation {val_labels[i].split('/')[-1]}")
  X, y = val_data.__getitem__(i)
  X = torch.Tensor(np.array([X.astype(np.float16)])).to(device)
  target_label = torch.Tensor(np.array([y.astype(np.float16)])).to(device)
  logits = model(X)
  mask = logits.cpu().detach().numpy()
  predicted_label = torch.max(logits[0], 0)[1].numpy()

  # compute classification metrics (dice coeff and area under ROC curve)
  (
    true_negative,
    false_positive,
    false_negative,
    true_positive,
  ) = confusion_matrix(
    target_label.reshape(-1), predicted_label.reshape(-1)
  ).ravel()
  dice_coeff = (
    2
    * true_positive
    / ((true_positive + false_positive) + (true_positive + false_negative))
  )
  print("dice is", dice_coeff)

  false_positive_rate, true_positive_rate, _ = roc_curve(
    target_label.reshape(-1), predicted_label.reshape(-1)
  )
  roc_auc = auc(false_positive_rate, true_positive_rate)
  print("ROC area under curve is", roc_auc)

  prec, recall, _ = precision_recall_curve(target_label.reshape(-1), predicted_label.reshape(-1), pos_label=1)
  prec_recall_auc = auc(prec, recall)

  print("Precision-recall area under curve is", prec_recall_auc)
  print()

