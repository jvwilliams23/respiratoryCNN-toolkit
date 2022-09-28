import argparse
import os
from glob import glob
from sys import exit

import hjson
#import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchio as tio
from sklearn.model_selection import train_test_split
from torch.utils import data

from data import *
from enet import ENet
from model import UNet

def get_inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "-r",
    "--resume_training",
    default=False,
    action="store_true",
    help="load pre-trained model and resume training",
  )
  return parser.parse_args()

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

args = get_inputs()
torch.backends.cudnn.benchmark = True

# -Open and load configurations.
with open("trainconfig.json") as f:
  config = hjson.load(f)

device = torch.device("cpu")

if os.path.isfile("train_cases.txt") and os.path.isfile("val_cases.txt"):
  train_scans, train_labels = convert_text_to_filenames("train_cases.txt")
  val_scans, val_labels = convert_text_to_filenames("val_cases.txt")
else:
  image_file_list, label_file_list = convert_text_to_filenames("cnn_training_cases.txt")
  train_scans, val_scans, train_labels, val_labels = train_test_split(
    image_file_list,
    label_file_list,
    test_size=config["train3d"]["validation_size"],
    shuffle=True,
  )

print("train scans: ")
for train_scan_i, train_labels_i in zip(train_scans, train_labels):
  print(train_scan_i, train_labels_i)
print("validation scans: ")
for val_scan_i, val_labels_i in zip(val_scans, val_labels):
  print(val_scan_i, val_labels_i)

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
  print("Training UNet")
  model = UNet(
    1,
    config["train3d"]["n_classes"],
    config["train3d"]["start_filters"],
    bilinear=False,
  ).to(device)
  model_string = "unet-"  # string to use for saving the model
  if args.resume_training:
    model.load_state_dict(torch.load(f"./{model_string}model.pt"))
elif config["train3d"]["model"].lower() == "enet":
  print("Training ENet")
  model = ENet(config["train3d"]["n_classes"]).to(device)
  model_string = "enet"  # string to use for saving the model
else:
  print("Unrecognised model. Exiting.")
  exit()
criterion = utils.lossTang2019
# gamma in Focal loss to increase weighting for voxel imbalance
gamma = config["train3d"]["gamma_in_loss"]
optimizer = optim.Adam(model.parameters(), lr=config["train3d"]["lr"])
batch_size = config["train3d"]["batch_size"]
epochs = config["train3d"]["epochs"]
train_size = len(train_labels)
val_size = len(val_labels)
print("End of data adquisition")

#print(f"training with {config['train3d']}")

# from utils import EarlyStopping
# initialise object
PATIENCE = 10
early_stopping = utils.EarlyStopping(
  patience=PATIENCE, verbose=True, path=f"{model_string}model.pt"
)

# augmentation pipeline for images
rotation = tio.RandomAffine(degrees=45)
transforms=(rotation, tio.RandomFlip(axes=('LR',)))
transform=tio.Compose(transforms)

def augment_image_data(scan, mask):
  """
    Apply augmentation as defined in global torchio transforms
    to both image (CT) and label (segmentation)
  """
  #transform requires a 4D tensor so a new axis is added
  scan=scan[np.newaxis,:].astype(np.float64)
  mask=mask[np.newaxis,:].astype(np.int32)
  #print("mask new axis",mask.shape)
  # subject object, necessary for the augmentations to be consistent between mask and scan
  scan=tio.Image(tensor=scan)
  mask=tio.Image(tensor=mask)
  subject=tio.Subject(image=scan,segmentation=mask)
  transformed = transform(subject)
  #access the image and segmentation as numpy arrays and then convert type to float16 and int8
  #transformed.plot()
  return (transformed["image"].numpy().astype(np.float16), transformed["segmentation"].numpy().astype(np.int8))

aug_iters = config["train3d"]["aug_iters"]
model.train()

NUM_BATCHES_IN_EPOCH = train_size + aug_iters

for epoch in range(epochs):
  """if epoch > 30:
  stop_training()"""
  epoch_loss = 0
  for i in range(0, NUM_BATCHES_IN_EPOCH, batch_size):
    losses = torch.ones(config["train3d"]["n_classes"])
    batch_loss = 0
    # when iterated through all training data, augment data
    if i >= train_size:
      augID = np.random.randint(
        train_size - batch_size - 1
      )  # -pick random ID to augment
      print("Augmenting patient", augID)
      batch = np.array(
        [train_data.__getitem__(j)[0] for j in range(augID, augID + batch_size)]
      ).astype(np.float16)
      labels = np.array(
        [train_data.__getitem__(j)[1] for j in range(augID, augID + batch_size)]
      ).astype(np.int8)
      # augment segmentation and scan using the same random state
      batch[0][0], labels[0][0] = augment_image_data(batch[0][0], labels[0][0])
    else:
      # read training image ('batch') and label from train_data class
      batch = np.array(
        [train_data.__getitem__(j)[0] for j in range(i, i + batch_size)]
      ).astype(np.float16)
      labels = np.array(
        [train_data.__getitem__(j)[1] for j in range(i, i + batch_size)]
      ).astype(np.int8)

    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)
    batch.requires_grad = True
    labels.requires_grad = True
    optimizer.zero_grad()
    # get probabilities for training data
    logits = model(batch)
    # get loss in training data for each class in sample
    for c, channel in enumerate(losses):
      losses[c] = criterion(
        torch.nn.functional.softmax(logits, dim=1)[0][c], labels, c, gamma=gamma
      )
    print("LOSSES ARE", losses)
    loss = torch.mean(losses)
    loss.backward()  # retain_graph=True)
    optimizer.step()
    print(
      "Epoch {} Batch {} mean loss : {}".format(
        epoch + 1,
        (i + 1) % (train_size + aug_iters),
        loss.item() / batch_size,
      )
    )
    epoch_loss += loss / batch_size
    del batch
    del labels
    del logits
    torch.cuda.empty_cache()
  # calculate validation loss after last training sample in epoch
  print("Calculating validation loss")
  val_loss = 0
  for val_iter, _ in enumerate(val_data):
    print("Number of validation iterations", len(val_data))
    losses = torch.ones(config["train3d"]["n_classes"])
    batch = np.array(
      [
        val_data.__getitem__(j)[0]
        for j in range(val_iter, val_iter + batch_size)
      ]
    ).astype(np.float16)
    labels = np.array(
      [
        val_data.__getitem__(j)[1]
        for j in range(val_iter, val_iter + batch_size)
      ]
    ).astype(np.int8)

    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)
    print("val_iter", val_iter)
    # calculate probabilities for validation set
    logits = model(batch)
    # loss for each class in validation sample i
    for c, channel in enumerate(losses):
      losses[c] = criterion(
        torch.nn.functional.softmax(logits, dim=1)[0][c],
        labels,
        c,
        gamma=gamma,
      )
    loss = torch.mean(losses)
    print("LOSSES ARE", losses)
    print(
      "Epoch {} Batch {} mean loss : {}".format(
        epoch + 1,
        (val_iter + 1) % (len(val_data)),
        loss.item() / batch_size,
      )
    )
    val_loss += loss.item()
    # clean up to preserve RAM
    del batch
    del labels
    del logits
    del losses
    torch.cuda.empty_cache()
  val_loss /= val_size
  early_stopping(val_loss, model)
  print("\n # Validation Loss : ", val_loss)
  if early_stopping.early_stop:
    print("Early stopping")
    break
  print("\n")
