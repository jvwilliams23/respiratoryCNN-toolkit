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

To segment the dataset defined using the first three parameters in path within config.json (see Getting started) run python segment.py which will output the segmentations to the directory ./segmentations.

Code within utilities uses the structure: 
├── modelName : model name used for training
│   ├── rocresults : rocresults for modelName
│   ├── results : results for modelName (DC,AUC,fbeta)
│   ├── logTrain : training log file for modelName
│   └── template3948 : template mesh for morphing new landmarks to create a new airway mesh

boxplotDICEcoeff.py : plots DICECoeff and other pieces of data. To process models use airCAE1NEW=getfield("modelName") where modelName is the same as the model directory. These can be lined next to each other to get an array of the results: 
airCA5=getfield("airCA5")
airCAE1=getfield("airCAE1")
airCAE3=getfield("airCAE3")
airCAE5=getfield("airCAE5")
airCAR1=getfield("airCAR1")
To plot a different field to DC use airCA1=getfield("modelName","AUC"). Run using python boxplotDICEcoeff.py

plotLogTrain.py : plots model residuals during training. To plot for modelName use modelName=airwayvalidationdata("modelName/logTrain"). 

outlier.py: outputs datapoint outliers based on the following criteria 
𝑖𝑓 𝐷𝐶𝑖≤ 𝐷𝑐̅̅̅−2𝜎 𝑂𝑅( 𝑞1−𝑞2)𝐼,
where 𝑖 is the DICE for the 𝑖𝑡ℎ segmented scan, 𝐼 is the interquartile range, 𝑞 represents quartile range, 𝐷𝑐̅̅̅ is the average dice coefficient and 𝜎 is the standard deviation of the DICE.
To run the script edit lines:
modelName=outliers(modelName,"modelName") and run using python outsider.py. This script outputs a csv of outliers. 

## Built With
matplotlib
numpy 
pandas 

python 3.9
## Get Help
- Contact me on my-email@email.com

## References
List any key literature this project is based on.

## Acknowledgements
Acknowledge the Open-Source projects / people that you've included in your solution.
