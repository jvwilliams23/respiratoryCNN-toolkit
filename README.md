# Title of Your Project 

#### Description of your project

## Table of content

- [Motivation](#motivation)
- [**Getting Started**](#getting-started)
- [Built With](#built-with)
- [Get Help](#get-help)
- [References](#references)
- [Acknowledgments](#acknowledgements)

## Motivation
Aim and motivation of this project.

## Getting Started
Describe how to get started here.

### Usage
Describe how you use it here.
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

## Get Help
- Contact me on my-email@email.com

## References
List any key literature this project is based on.

## Acknowledgements
Acknowledge the Open-Source projects / people that you've included in your solution.
