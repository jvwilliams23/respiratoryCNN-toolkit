import numpy as np 
import matplotlib.pyplot as plt 
from math import floor
import matplotlib.ticker as plticker
import re
import os 
import pandas as pd
def gendicecoef(fname):
  f=open(fname,"r")
  lines=f.readlines()
  f.close()
  train = []
  val = []
  for line in lines:
      #-filter training and validation sets
    if line[0]=="t":
      train.append(float(line.split()[1]))
    elif line[0]=="v":
      val.append(float(line.split()[1]))
  train = np.array(train)
  val = np.array(val)
  return([train,val])

def getvaltrain(key):
    #determine if data is validation or training data S
    scans=[]
    gather=False
    trainscan=re.compile("train scans:.*?")
    valscan=re.compile("validation scans:.*?")
    datagetend=re.compile("train_data OK.*?")
    logFile=str(key[1:]) + os.sep + "logTrain"
    if key[0] == 'v':
        with open(logFile) as file:
            for line in file:
                if trainscan.match(line):
                    gather=False
                    continue
                if valscan.match(line):
                    gather=True
                    continue
                if datagetend.match(line):
                    return scans
                if gather==True:
                    scans.append(line.replace('\n',''))
    elif key[0] == 't':
        with open(logFile) as file:
            for line in file:
                if trainscan.match(line):
                    gather=True
                    continue
                if valscan.match(line):
                    gather=False
                    continue
                if datagetend.match(line):
                    return scans
                if gather==True:
                    scans.append(line.replace('\n',''))
    return(scans)

def getfield(model,field="DC"):
      #get list of training and validation scans
      valscans_=getvaltrain('v'+model)
      trainscans_=getvaltrain('t'+model)
      df=pd.read_csv(model+os.sep+"results.csv")
      valscans=[]
      trainscans=[]
      for it in valscans_:
            valscans.append(df[it.replace(".mhd","")[-4:]+"v "+field])
      for it in trainscans_:
            trainscans.append(df[it.replace(".mhd","")[-4:]+"t "+field])
      trainscans = np.array(trainscans)
      valscans = np.array(valscans)
      return([trainscans.flatten(),valscans.flatten()])

  

def writetocsv(arr,header,fname="dcairCNN.csv"):
  import csv
  train=arr[0].tolist()
  train.insert(0,str(header+" train"))
  val=arr[1].tolist()
  val.insert(0,str(header+" validation"))
  trainavg=np.average(arr[0])
  valavg=np.average(arr[1])
  #train std
  trainstd=np.std(arr[0])
  valstd=np.std(arr[1])
  outstdrangetrain=[]
  outstdrangeval=[]
  for ele in arr[0]:
        if ele<trainavg-2*trainstd:
              outstdrangetrain.append(ele)
  for ele in arr[1]:
        if ele<valavg-2*valstd:
              outstdrangeval.append(ele)
  #format
  trainavgheader=str(header+" train average")
  validationavgheader=str(header+" validation average")
  trainavg=[trainavgheader,trainavg]
  valavg=[validationavgheader,valavg]
  print(valavg)
  #grep ID
  outstdrangetrainID=[]
  if len(outstdrangetrain)>0:
        for ele in outstdrangetrain:
              pattern=re.compile("(.*?)"+str(ele))
              with open(header+os.sep+"airwayDice.dat") as dat:
                for line in dat:
                      if pattern.match(line):
                            outstdrangetrainID.append(pattern.match(line).group(1))
  #format
  outstdrangetrainID.insert(0,"train out std range ID")
  #grep ID
  outstdrangevalID=[]
  if len(outstdrangeval)>0:
        for ele in outstdrangeval:
              pattern=re.compile("(.*?)"+str(ele))
              with open(header+os.sep+"airwayDice.dat") as dat:
                for line in dat:
                      if pattern.match(line):
                            outstdrangevalID.append(pattern.match(line).group(1))
  #format
  outstdrangevalID.insert(0,"val out std range ID")
        
  outstdrangetrain.insert(0,str(header+" train out std range"))
  outstdrangeval.insert(0,str(header+" validation out std range"))
  #Q1-1.5*IQR
  q3train,q1train=np.percentile(arr[0], [75,25])
  q3val,q1val=np.percentile(arr[1], [75,25])
  iqrtrain=q3train-q1train
  iqrval=q3val-q1val
  outqrangetrain=[]
  outqrangeval=[]
  for ele in arr[0]:
        if ele<q1train-1.5*iqrtrain:
              outqrangetrain.append(ele)
  for ele in arr[1]:
        if ele<q1val-1.5*iqrval:
              outqrangeval.append(ele)
  ####################################
  #grep ID
  outqrangevalID=[]
  if len(outqrangeval)>0:
        for ele in outqrangeval:
              pattern=re.compile("(.*?)"+str(ele))
              with open(header+os.sep+"airwayDice.dat") as dat:
                for line in dat:
                      if pattern.match(line):
                            outqrangevalID.append(pattern.match(line).group(1))
  #format
  outqrangevalID.insert(0,"val out q range ID")
  #grep ID
  outqrangetrainID=[]
  if len(outqrangetrain)>0:
        for ele in outqrangetrain:
              pattern=re.compile("(.*?)"+str(ele))
              with open(header+os.sep+"airwayDice.dat") as dat:
                for line in dat:
                      if pattern.match(line):
                            outqrangetrainID.append(pattern.match(line).group(1))
  #format
  outqrangetrainID.insert(0,"train out q range ID")
  ###############################
  #format
  outqrangetrain.insert(0,str(header+" train outliers"))
  outqrangeval.insert(0,str(header+" validation outliers"))
  with open(fname, 'a', newline='') as myfile:
    csvWriter = csv.writer(myfile,delimiter=',')
    csvWriter.writerows([train,trainavg,outstdrangetrainID,outstdrangetrain,outqrangetrainID,outqrangetrain,
    val,valavg,outstdrangevalID,outstdrangeval,outqrangevalID,outqrangeval])


      
"""
f=open("airCA3/airwayDice.dat","r")
lines=f.readlines()
f.close()

train = []
val = []
for line in lines:
  #-filter training and validation sets
  if line[0]=="t":
    train.append(float(line.split()[1]))
  elif line[0]=="v":
    val.append(float(line.split()[1]))

#-convert to array
train = np.array(train)
val = np.array(val)
diceMin = floor( np.min( np.hstack((train,val)) ) * 10)/10
"""
"""
airCA1=gendicecoef("airCA1/airwayDice.dat")
airCA3=gendicecoef("airCA3/airwayDice.dat")
airCA5=gendicecoef("airCA5/airwayDice.dat")
airCAE1=gendicecoef("airCAE1/airwayDice.dat")
airCAE3=gendicecoef("airCAE3/airwayDice.dat")
airCAE5=gendicecoef("airCAE5/airwayDice.dat")
airCAR1=gendicecoef("airCAR1/airwayDice.dat")
airCAR3=gendicecoef("airCAR3/airwayDice.dat")
airCAR5=gendicecoef("airCAR5/airwayDice.dat")
"""
'''
#dice for airCAR5 excessive augmentations
airCAR5100=getfield("airCAR5100")
fig, axs = plt.subplots(2,dpi=300,constrained_layout=True)
print("average train dice coefficient during validation: airCAR5100",round(np.average(airCAR5100[1]),3))
print("average train dice coefficient during training: airCAR5100",round(np.average(airCAR5100[0]),3))
#axes = plt.gca()
#y_pos=0.835 #gamma r 3
#axes.set_ylim([0.83,None])
#0.88 and 0.83
second1=axs[0].secondary_yaxis('right')
second2=axs[1].secondary_yaxis('right')
loc = plticker.MultipleLocator(base=0.01)
axs[0].set(ylim=(0.86,None))
axs[1].set(ylim=(0.86,None))
axs[0].yaxis.set_major_locator(loc)
axs[1].yaxis.set_major_locator(loc)
second1.yaxis.set_major_locator(loc)
second2.yaxis.set_major_locator(loc)
second1.tick_params(axis='both', which='major', labelsize=7)
second2.tick_params(axis='both', which='major', labelsize=7)
axs[0].tick_params(axis='both', which='major', labelsize=7)
axs[1].tick_params(axis='both', which='major', labelsize=7)
#\u03B3=5 rotation
axs[0].boxplot((airCAR5100[0]),
 labels=["100 augmentations"])
axs[1].boxplot((airCAR5100[1]),
 labels=["100 augmentations"])
axs[0].set_ylabel("Dice coefficient", fontsize=10)
axs[1].set_ylabel("Dice coefficient", fontsize=10)
axs[0].set_title("Dice coefficient \u03B3=5 rotation during training")
axs[1].set_title("Dice coefficient \u03B3=5 rotation during validation")

plt.show()
'''
"""
writetocsv(airCA1,"airCA1")
writetocsv(airCA3,"airCA3")
writetocsv(airCA5,"airCA5")
writetocsv(airCAE1,"airCAE1")
writetocsv(airCAE3,"airCAE3")
writetocsv(airCAE5,"airCAE5")
writetocsv(airCAR1,"airCAR1")
writetocsv(airCAR3,"airCAR3")
writetocsv(airCAR5,"airCAR5")
"""
airCAE1NEW=getfield("airCAE1NEW")
airCAE5NEW=getfield("airCAE5NEW")
airCA1=getfield("airCA1")
airCA3=getfield("airCA3")
airCA5=getfield("airCA5")
airCAE1=getfield("airCAE1")
airCAE3=getfield("airCAE3")
airCAE5=getfield("airCAE5")
airCAR1=getfield("airCAR1")
airCAR3=getfield("airCAR3")
airCAR5=getfield("airCAR5")
airCAR5100=getfield("airCAR5100")
airCAR550=getfield("airCAR5 50")

"""
airCA1=getfield("airCA1","AUC")
airCA3=getfield("airCA3","AUC")
airCA5=getfield("airCA5","AUC")
airCAE1=getfield("airCAE1","AUC")
airCAE3=getfield("airCAE3","AUC")
airCAE5=getfield("airCAE5","AUC")
airCAR1=getfield("airCAR1","AUC")
airCAR3=getfield("airCAR3","AUC")
airCAR5=getfield("airCAR5","AUC")
"""

print("average train dice coefficient: airCA3",round(np.average(airCA3[0]),3))
print("average train dice coefficient: airCAE3",round(np.average(airCAE3[0]),3))
print("average train dice coefficient: airCAR3",round(np.average(airCAR3[0]),3))
print("average train dice coefficient: airCA1",round(np.average(airCA1[0]),3))
print("average train dice coefficient: airCAE1",round(np.average(airCAE1[0]),3))
print("average train dice coefficient: airCAR1",round(np.average(airCAR1[0]),3))
print("average train dice coefficient: airCAE5",round(np.average(airCAE5[0]),3))
print("average train dice coefficient: airCA5",round(np.average(airCA5[0]),3))
print("average train dice coefficient: airCAR5",round(np.average(airCAR5[0]),3))
print("average train dice coefficient: airCAR5100",round(np.average(airCAR5100[0]),3))
print("average train dice coefficient: airCAR550",round(np.average(airCAR550[0]),3))
print("average val dice coefficient: airCA3",round(np.average(airCA3[1]),3))
print("average val dice coefficient: airCAE3",round(np.average(airCAE3[1]),3))
print("average val dice coefficient: airCAR3",round(np.average(airCAR3[1]),3))
print("average val dice coefficient: airCA1",round(np.average(airCA1[1]),3))
print("average val dice coefficient: airCAE1",round(np.average(airCAE1[1]),3))
print("average val dice coefficient: airCAR1",round(np.average(airCAR1[1]),3))
print("average val dice coefficient: airCA5",round(np.average(airCA5[1]),3))
print("average val dice coefficient: airCAE5",round(np.average(airCAE5[1]),3))
print("average val dice coefficient: airCAR5",round(np.average(airCAR5[1]),3))
print("average val dice coefficient: airCAR5100",round(np.average(airCAR5100[1]),3))
print("average val dice coefficient: airCAR550",round(np.average(airCAR550[1]),3))

"""-------------------------------------------------------------------------------"""
print("average train dice coefficient: airCAE1NEW",round(np.average(airCAE1NEW[0]),3))
print("average val dice coefficient: airCAE1NEW",round(np.average(airCAE1NEW[1]),3))

print("average train dice coefficient: airCAE5NEW",round(np.average(airCAE5NEW[0]),3))
print("average val dice coefficient: airCAE5NEW",round(np.average(airCAE5NEW[1]),3))

print("--------------------------------", airCAR5)
plt.close()
fig, axs = plt.subplots(2,dpi=220,constrained_layout=True)
#axes = plt.gca()
#y_pos=0.835 #gamma r 3
#axes.set_ylim([0.83,None])
#0.88 and 0.83
#second1=axs[0].secondary_yaxis('right')
#second2=axs[1].secondary_yaxis('right')
loc = plticker.MultipleLocator(base=0.02)
axs[0].set(ylim=(0.828,None))
axs[1].set(ylim=(0.828,None))
axs[0].yaxis.set_major_locator(loc)
axs[1].yaxis.set_major_locator(loc)
#second1.yaxis.set_major_locator(loc)
#second2.yaxis.set_major_locator(loc)
#second1.tick_params(axis='both', which='major', labelsize=10)
#second2.tick_params(axis='both', which='major', labelsize=10)
axs[0].tick_params(axis='both', which='major', labelsize=13)
axs[1].tick_params(axis='both', which='major', labelsize=13)
#],airCAR550[1],airCAR5100[1]),

axs[0].boxplot((airCA1[0],airCA3[0],airCA5[0],airCAE1[0],airCAE3[0],airCAE5[0],airCAR1[0],airCAR3[0],airCAR5[0],airCAE1NEW[0],airCAE5NEW[0]),
 labels=["\u03B3=1","\u03B3=3","\u03B3=5","\u03B3=1 p","\u03B3=3 p",
 "\u03B3=5 p","\u03B3=1 r","\u03B3=3 r","\u03B3=5 r*","\u03B3=1 e","\u03B3=5 e"])
axs[1].boxplot((airCA1[1],airCA3[1],airCA5[1],airCAE1[1],airCAE3[1],airCAE5[1],airCAR1[1],airCAR3[1],airCAR5[1],airCAE1NEW[1],airCAE5NEW[1]),
 labels=["\u03B3=1","\u03B3=3","\u03B3=5","\u03B3=1 p","\u03B3=3 p",
 "\u03B3=5 p","\u03B3=1 r","\u03B3=3 r","\u03B3=5 r*","\u03B3=1 e","\u03B3=5 e"])

axs[0].set_ylabel("Dice coefficient", fontsize=14)
axs[1].set_ylabel("Dice coefficient", fontsize=14)
axs[0].set_title("Dice coefficient training",fontsize=14)
axs[1].set_title("Dice coefficient validation",fontsize=14)
plt.show()
