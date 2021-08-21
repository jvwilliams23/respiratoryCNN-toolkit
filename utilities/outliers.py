import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
def getvaltrain(key):
    #determine if data is validation or training data 
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
      valID=[]
      trainID=[]
      df=pd.read_csv(model+os.sep+"results.csv")
      valscans=[]
      trainscans=[]
      for it in valscans_:
            valID.append(it.replace(".mhd","")[-4:])
            valscans.append(df[it.replace(".mhd","")[-4:]+"v "+field])
      for it in trainscans_:
            trainID.append(it.replace(".mhd","")[-4:])
            trainscans.append(df[it.replace(".mhd","")[-4:]+"t "+field])
      trainscans = np.array(trainscans)
      valscans = np.array(valscans)
      return([trainscans.flatten(),valscans.flatten(),trainID,valID])

def outliers(arr,header):
  outstdrangetrainID=[]
  outstdrangevalID=[]
  outqrangetrainID=[]
  outqrangevalID=[]
  outlier_dict={header:{"SD":{
   "train":[[],[]],
   "val":[[],[]]
  },"Q":{
    "train":[[],[]],
   "val":[[],[]]
  }
  }}
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
  for ele,it in zip(arr[0],arr[2]):
        if ele<trainavg-2*trainstd:
              outstdrangetrain.append(ele)
              outstdrangetrainID.append(it)
  for ele,it in zip(arr[1],arr[3]):
        if ele<valavg-2*valstd:
              outstdrangeval.append(ele)
              outstdrangevalID.append(it)
  #format
  trainavgheader=str(header+" train average")
  validationavgheader=str(header+" validation average")
  trainavg=[trainavgheader,trainavg]
  valavg=[validationavgheader,valavg]
  #format
  #Q1-1.5*IQR
  q3train,q1train=np.percentile(arr[0], [75,25])
  q3val,q1val=np.percentile(arr[1], [75,25])
  iqrtrain=q3train-q1train
  iqrval=q3val-q1val
  outqrangetrain=[]
  outqrangeval=[]
  for ele,it in zip(arr[0],arr[2]):
        if ele<q1train-1.5*iqrtrain:
              outqrangetrain.append(ele)
              outqrangetrainID.append(it)
  for ele,it in zip(arr[1],arr[3]):
        if ele<q1val-1.5*iqrval:
              outqrangeval.append(ele)
              outqrangevalID.append(it)
  ####################################
  
  #format
  if len(outqrangetrain)>0:
    outlier_dict[header]["Q"]["train"][0].extend(outqrangetrain) #copy this format
    outlier_dict[header]["Q"]["train"][1].extend(outqrangetrainID) #copy this format
  if len(outqrangeval)>0:
    outlier_dict[header]["Q"]["val"][0].extend(outqrangeval) #copy this format
    outlier_dict[header]["Q"]["val"][1].extend(outqrangevalID) #copy this format
  if len(outstdrangetrain)>0:
    outlier_dict[header]["SD"]["train"][0].extend(outstdrangetrain) #copy this format
    outlier_dict[header]["SD"]["train"][1].extend(outstdrangetrainID) #copy this format
    #val IDs are same 
  if len(outstdrangeval)>0:
    outlier_dict[header]["SD"]["val"][0].extend(outstdrangeval) #copy this format
    outlier_dict[header]["SD"]["val"][1].extend(outstdrangevalID) #copy this format
  return outlier_dict
  ###############################
  #format
  """
  outqrangetrain.insert(0,str(header+" train outliers"))
  outqrangeval.insert(0,str(header+" validation outliers"))
  with open(fname, 'a', newline='') as myfile:
    csvWriter = csv.writer(myfile,delimiter=',')
    csvWriter.writerows([train,trainavg,outstdrangetrainID,outstdrangetrain,outqrangetrainID,outqrangetrain,
    val,valavg,outstdrangevalID,outstdrangeval,outqrangevalID,outqrangeval])
"""
def writedict(data):
  import csv
  import itertools
  #set headers to use in all 4 sheets 
  with open('outliersFinal.csv',"a",newline="") as f:
      w = csv.writer( f )
      for key in data:
          it=data[key]["SD"]["train"][0]
          w.writerow([key]+["SD"]+["train"]+it)
          it=data[key]["SD"]["train"][1]
          w.writerow([key]+["SD"]+["train"]+it)
          it=data[key]["SD"]["val"][0]
          w.writerow([key]+["SD"]+["val"]+it)
          it=data[key]["SD"]["val"][1]
          w.writerow([key]+["SD"]+["val"]+it)
          it=data[key]["Q"]["train"][0]
          w.writerow([key]+["Q"]+["train"]+it)
          it=data[key]["Q"]["train"][1]
          w.writerow([key]+["Q"]+["train"]+it)
          it=data[key]["Q"]["val"][0]
          w.writerow([key]+["Q"]+["val"]+it)
          it=data[key]["Q"]["val"][1]
          w.writerow([key]+["Q"]+["val"]+it)

    #for it in range(data):
    
    #set rows
airCA1=getfield("airCA1") #field as a dict or dataframe
print(airCA1)
headers=np.concatenate((airCA1[2],airCA1[3]),axis=0).tolist()
headers.insert(0,"model")
import csv
with open('segmentationqualityFinal.csv','w',newline="") as f:
    w= csv.writer(f)
    w.writerow(headers)
 
airCA1_out=outliers(airCA1,"airCA1")
print(airCA1_out)
airCA3=getfield("airCA3")
airCA3_out=outliers(airCA3,"airCA3")
print(airCA3_out)
airCA5=getfield("airCA5")
airCA5_out=outliers(airCA5,"airCA5")
print(airCA5_out)
airPW1=getfield("airCAE1")
airPW1_out=outliers(airPW1,"airPW1")
print(airPW1_out)
airPW3=getfield("airCAE3")
airPW3_out=outliers(airPW3,"airPW3")
print(airPW3_out)
airPW5=getfield("airCAE5")
airPW5_out=outliers(airPW5,"airPW5")
print(airPW5_out)
airCAR1=getfield("airCAR1")
airCAR1_out=outliers(airCAR1,"airCAR1")
print(airCAR1_out)
airCAR3=getfield("airCAR3")
airCAR3_out=outliers(airCAR3,"airCAR3")
print(airCAR3_out)
airCAR5=getfield("airCAR5")
airCAR5_out=outliers(airCAR5,"airCAR5")
print(airCAR5_out)

'''-------------------'''
airCAE1=getfield("airCAE1NEW")
airCAE1_out=outliers(airCAE1,"airCAE1")
print(airCAE1_out)
airCAE5=getfield("airCAE5NEW")
airCAE5_out=outliers(airCAE5,"airCAE5")
print(airCAR5_out)

writedict(airCA1_out)
writedict(airCA3_out)
writedict(airCA5_out)
writedict(airPW1_out)
writedict(airPW3_out)
writedict(airPW5_out)
writedict(airCAR1_out)
writedict(airCAR3_out)
writedict(airCAR5_out)
"""---------------"""
writedict(airCAE1_out)
writedict(airCAE5_out)

"""
objects = ('4301', '2456', '8748', '7734', '0723', '2488','5837','2523','1437','3123')
y_pos = np.arange(len(objects))
performance = [11,7,2,1,11,8,7,1,2,2]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('outlier frequency (maximum of 36)')
plt.xlabel('case number')
plt.title('dice coefficient outliers')
plt.savefig("outliers.jpg",dpi=300)
#plt.show()
#max 36
"""
"""
number of outliers for each model
"""
