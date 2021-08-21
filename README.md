Automated airway segmentation from chest CT scans using deep learning

Table of content:
- Motivation
- Getting Started:
- Built with  
- Get Help
- References
- Acknowledgments

## Motivation
This project aimed to train a machine learning algorithm capable of producing airway segmentations. This repo contains the relevant code to allow anyone to reproduce the training and segment a chest ct scan (hardware providing).

## Getting Started
Describe how to get started here.
 The data used in this project can be defined using the config.json file. Starting from the top: within path the directories for the ct scans, masks, labelled_list represent data directories used within model.py (training) and evalDICE.py (model evaluation). Whilst test_scans,unseen_masks and unseen_scans also within path represent data directories for unseen datasets used within evalDiceTest.py (unseen dataset model evaluation). Within train3d and segment3d the parameters used for training the model are set. 

### Usage
Scripts don't take optional parameters: all parameters are set in the config.json. 
To train the model use python train.py > logTrain. It is important to redirect output to logTrain since this is used to determine the current performance of the classifier. This is used in to determine when to stop training by taking a decrease of 5% in the validation DICE over 30 epochs as a cut off point for stopping the model. 

Once the model is trained it can be evaluated using evalDICE.py (python evalDICE.py) which outputs the results in files results.csv and rocresults.csv. Results are shown in the following format: 
3948t DC    0645v DC
0.960152855 0.952423835
where the number (3948) represents the unique image ID, t training dataset (full list of train validation split given in logTrain), v validation dataset, DC dice coefficient and the number below gives the respective score.

To segment the dataset defined using the first three parameters in path within config.json (see Getting started) run python segment.py which will output the segmentations to the directory ./segmentations
```bash
python codename.py --arg1 foo --arg2 bar
```
Or import to your class by
```python
from codename import CodeClass

a = CodeClass(arg1=foo,
              arg2=bar,
              arg3=cat)
```
If any particular naming structure is required for input data or folders, give here.
Here is an example from one of my repos.
```bash
├── allLandmarks : combined airways and lung lobes landmarks. Some 'landmarkIndex' files that 
|                  assign return the index in numpy array for a specific part (for slicing 
|                  i.e. airway_lms[skeleton_ids])
├── landmarks
│   ├── manual-jw : manually selected landmarks of branch points for first two bifurcations
│   └── manual-jw-diameterFromSurface : airway landmarks with interpolated points along branch
|                                       segment and points along the surface to include 
|                                       diameter in shape model.
├── outputLandmarks : example of landmarks output from fitting a shape model to unseen data.
├── segmentations : ground truth segmentations for a few datasets.
│   ├── airwaysForRadiologistWSurf
│   │   ├── 0576
│   │   ├── 3948
│   │   ├── 5837
│   │   ├── 8684
│   │   └── 8865
│   └── template3948 : template mesh for morphing new landmarks to create a new airway mesh
└── skelGraphs : networkx graphs with full skeleton and skeleton with branch points only.
```

Description of scripts

Toy example scripts are:
```bash
├── abc.py : create XYZ
├── abc_post.py : post-processing script to get XYZ
├── plot_abc.py : plot output of post-processing script
```

real example :
```bash
├── formatLandmarks_diameterFromSurface.py : create airway landmarks with interpolated points 
|                                             on branch segments and points on surface of mesh
├── gtToReconstructionLMdiff.py : post-processing script to get distance between reconstructed 
|                                 landmarks and some morphological parameters 
|                                  (branch length + angle)
├── tschirrenSkeletonise.py : skeletonise a mhd file and clean up with networkx.
                              Loosely based on Tschirren et al. (2004)
```


## Built With
List which key libraries you've used here.
python 3.9
## Get Help
- Contact me on my-email@email.com

## References
List any key literature this project is based on.

## Acknowledgements
Acknowledge the Open-Source projects / people that you've included in your solution.
