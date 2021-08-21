import re
import numpy as np
from statistics import mean
import sys
from collections import defaultdict
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker
def airwayvalidationdata(logfile):
    logdata=[]
    logdata_index=[]
    validationsection=False
    iterator=0
    patternDICE=re.compile("DICE IS tensor(.*?)(\d*$)(.*?)")
    validationbegin=re.compile("(.*?)Calculating validation loss(.*?)")
    validationend=re.compile("(.*?)Validation Loss(.*?)")
    with open(logfile) as file:
        for line in file:
            if validationbegin.match(line):
                validationsection=True
                continue
            if validationend.match(line):
                iterator=iterator+1
                validationsection=False
                continue
            if validationsection==True:
                if patternDICE.match(line):
                    logdata.append(float((re.sub('[^\d\.]','',line))))
                    logdata_index.append(iterator)
                    continue
    result = []
    temp = defaultdict(list)
    logdata=list(zip(logdata[1::2],logdata_index[1::2]))
    for ele in logdata: 
        temp[ele[1]].append(ele[0])
    for k,v in temp.items():
        result.append((np.average(np.array(v))).tolist())
    #make into a matrix for each epoch 
    #also could use lambda group() functionality
    return([(np.arange(start=0, stop=len(result),step=1)).tolist(),result])

def backgroundvalidationdata(logfile):
    logdata=[]
    logdata_index=[]
    validationsection=False
    iterator=0
    patternDICE=re.compile("DICE IS tensor(.*?)(\d*$)(.*?)")
    validationbegin=re.compile("(.*?)Calculating validation loss(.*?)")
    validationend=re.compile("(.*?)Validation Loss(.*?)")
    with open(logfile) as file:
        for line in file:
            if validationbegin.match(line):
                validationsection=True
                continue
            if validationend.match(line):
                iterator=iterator+1
                validationsection=False
                continue
            if validationsection==True:
                if patternDICE.match(line):
                    logdata.append(float((re.sub('[^\d\.]','',line))))
                    logdata_index.append(iterator)
                    continue
    result = []
    temp = defaultdict(list)
    del(logdata[1::2])
    del(logdata_index[1::2])
    logdata=list(zip(logdata,logdata_index))
    for ele in logdata: 
        temp[ele[1]].append(ele[0])
    for k,v in temp.items():
        result.append((np.average(np.array(v))).tolist())
    #make into a matrix for each epoch 
    #also could use lambda group() functionality
    return([(np.arange(start=0, stop=len(result),step=1)).tolist(),result])
def airwayvalidationdataold(logfile):   
    logdata=[]
    logdata_index=[]
    validationsection=False
    iterator=int(0)
    patternDICE=re.compile("DICE IS tensor(.*?)(\d*$)(.*?)")
    validationbegin=re.compile("(.*?)Calculating validation loss(.*?)")
    validationend=re.compile("(.*?)Validation Loss(.*?)")
    epochnum = re.compile("Epoch ([0-9]{1,3})")
    with open(logfile) as file:
        for line in file:
            if validationbegin.match(line):
                validationsection=True
                continue
            if validationend.match(line):
                validationsection=False
                continue
            if epochnum.match(line):
                iterator=int(epochnum.match(line).group(1))
            if validationsection==True:
                if patternDICE.match(line):
                    logdata.append(float((re.sub('[^\d\.]','',line))))
                    logdata_index.append(iterator)
                    continue
    result = []
    temp = defaultdict(list)
    logdata=list(zip(logdata[1::2],logdata_index[1::2]))
    for ele in logdata: 
        temp[ele[1]].append(ele[0])
    for k,v in temp.items():
        result.append((np.average(np.array(v))).tolist())
    #make into a matrix for each epoch 
    #also could use lambda group() functionality
    return([(np.arange(start=0, stop=len(result),step=1)).tolist(),result])

def backgroundvalidationdataold(logfile):   
    logdata=[]
    logdata_index=[]
    validationsection=False
    iterator=int(0)
    patternDICE=re.compile("DICE IS tensor(.*?)(\d*$)(.*?)")
    validationbegin=re.compile("(.*?)Calculating validation loss(.*?)")
    validationend=re.compile("(.*?)Validation Loss(.*?)")
    epochnum = re.compile("Epoch ([0-9]{1,3})")
    with open(logfile) as file:
        for line in file:
            if validationbegin.match(line):
                validationsection=True
                continue
            if validationend.match(line):
                validationsection=False
                continue
            if epochnum.match(line):
                iterator=int(epochnum.match(line).group(1))
            if validationsection==True:
                if patternDICE.match(line):
                    logdata.append(float((re.sub('[^\d\.]','',line))))
                    logdata_index.append(iterator)
                    continue
    result = []
    temp = defaultdict(list)
    del(logdata[1::2])
    del(logdata_index[1::2])
    logdata=list(zip(logdata,logdata_index))
    for ele in logdata: 
        temp[ele[1]].append(ele[0])
    for k,v in temp.items():
        result.append((np.average(np.array(v))).tolist())
    #make into a matrix for each epoch 
    #also could use lambda group() functionality
    return([(np.arange(start=0, stop=len(result),step=1)).tolist(),result])

def predict(data,start,stop,num):
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(np.array(data[0]).reshape(-1,1).tolist(),np.array(data[1]).reshape(-1,1).tolist())
    xn=np.linspace(start,stop,num,dtype='int').reshape(-1,1).tolist()
    return(xn,model.predict(xn))

def plot(result,label="line"):
   return(plt.plot(result[0],result[1],label=label))

def check(file,log):
    with open(file,"w") as writefile:
        with open(log) as myfile:
            firstNlines=myfile.readlines()[1904:] #put here the interval you want
            for item in firstNlines:
                writefile.write("%s\n" % item)

"""
plt.title("Airway residuals")
plt.xlabel( 'Number of epochs',fontsize=15)
plt.ylabel('Residuals',fontsize=15)
airCA1=airwayvalidationdata("airCA1/logTrain")
airCA1,=plot(airCA1,"\u03B3=1")
airCA3=airwayvalidationdata("airCA3/logTrain")
airCA3,=plot(airCA3,"\u03B3=3")
airCA5=airwayvalidationdata("airCA5/logTrain")
airCA5,=plot(airCA5,"\u03B3=5")
airCAE1=airwayvalidationdata("airCAE1/logTrain")
airCAE1,=plot(airCAE1,"\u03B3=1 Elastic")
airCAE3=airwayvalidationdata("airCAE3/logTrain")
airCAE3,=plot(airCAE3,"\u03B3=3 Elastic")
airCAE5=airwayvalidationdata("airCAE5/logTrain")
airCAE5,=plot(airCAE5,"\u03B3=5 Elastic")
airCAR3=airwayvalidationdata("airCAR3/logTrain")
airCAR3,=plot(airCAR3,"\u03B3=3 Rotation")
airCAR5=airwayvalidationdata("airCAR5/logTrain")
airCAR5,=plot(airCAR5,"\u03B3=5 Rotation")
plt.legend(handles=[airCA1,airCA3,airCA5,airCAE1,airCAE3,airCAE5,airCAR3,airCAR5],loc='lower left')
#better as subplots
#plt.show()
plt.savefig("allresidual",dpi=300)
"""
'''
#main log
airCA1=airwayvalidationdata("airCA1/logTrain")
airCA3=airwayvalidationdata("airCA3/logTrain")
airCA5=airwayvalidationdata("airCA5/logTrain")
airCAE1=airwayvalidationdata("airCAE1/logTrain")
airCAE3=airwayvalidationdata("airCAE3/logTrain")
airCAE5=airwayvalidationdata("airCAE5/logTrain")
airCAR1=airwayvalidationdata("airCAR1/logTrain")
airCAR3=airwayvalidationdata("airCAR3/logTrain")
airCAR5=airwayvalidationdata("airCAR5/logTrain")
airCAR5100=airwayvalidationdata("airCAR5100/logTrain")
fig, axs = plt.subplots(3, 1,figsize=plt.figaspect(1.4),constrained_layout=True,sharex=True,sharey=True,dpi=300)#constrained_layout=True
fig.suptitle('Airway residuals')
loc = plticker.MultipleLocator(base=5.0)
loc2 = plticker.MultipleLocator(base=0.1)
axs[2].xaxis.set_major_locator(loc)
axs[2].yaxis.set_major_locator(loc2)
axs[2].tick_params(axis='both', which='major', labelsize=7)
axs[2].tick_params(axis='both', which='minor', labelsize=7)
axs[0].tick_params(axis='both', which='major', labelsize=7)
axs[1].tick_params(axis='both', which='major', labelsize=7)
#not obvious y axis needs more points and non sharex
axs[2].set_xlabel( 'Number of epochs',fontsize=16) #could put this in the figure title
axs[1].set_ylabel('Residuals',fontsize=16)
axs[0].set_title("\u03B3=1",fontsize=10)
airCA1,=axs[0].plot(airCA1[0],airCA1[1],label="no augmentation")
airCAE1,=axs[0].plot(airCAE1[0],airCAE1[1],label="elastic augmentation")
airCAR1,=axs[0].plot(airCAR1[0],airCAR1[1],label="rotational augmentation")
axs[0].legend([airCA1,airCAE1,airCAR1],["no augmentation","elastic","rotation"])
axs[1].set_title("\u03B3=3",fontsize=10)
airCA3,=axs[1].plot(airCA3[0],airCA3[1])
airCAE3,=axs[1].plot(airCAE3[0],airCAE3[1])
airCAR3,=axs[1].plot(airCAR3[0],airCAR3[1])
axs[1].legend([airCA3,airCAE3,airCAR3],["no augmentation","elastic","rotation"])

axs[2].set_title("\u03B3=5",fontsize=10)
airCA5,=axs[2].plot(airCA5[0],airCA5[1],label="no augmentation")
airCAE5,=axs[2].plot(airCAE5[0],airCAE5[1],label="elastic augmentation")
airCAR5,=axs[2].plot(airCAR5[0],airCAR5[1],label="rotational augmentation")
airCAR5100,=axs[2].plot(airCAR5100[0],airCAR5100[1],label="rotational augmentation")
axs[2].legend([airCA5,airCAE5,airCAR5,airCAR5100],["no augmentation","elastic","rotation"])
#excessive augmentations with airCAR5
plt.show()
'''
#airCAR1=airwayvalidationdata("airCAR1/logTrain")
#airCAR3=airwayvalidationdata("airCAR3/logTrain")
airCAR5=airwayvalidationdata("airCAR5/logTrain")
#airCAR5100=airwayvalidationdata("airCAR5100/logTrain")
airCAECancelled=airwayvalidationdata("failedairCAE3/logTrain")
plt.title("Validation curve",fontsize=19)
#airCAECancelled,=plt.plot(airCAECancelled[0],airCAECancelled[1])
airCAR5,=plt.plot(airCAR5[0],airCAR5[1])
#airCAR3,=plt.plot(airCAR3[0],airCAR3[1])
#airCAR5,=plt.plot(airCAR5[0],airCAR5[1])
#airCAR5100,=plt.plot(airCAR5100[0],airCAR5100[1])
plt.xlim([0,75])
plt.legend([airCAR5],["Model"])
plt.xlabel("Number of epochs",fontsize=19)
plt.ylabel("Residuals",fontsize=19)
plt.savefig(fname="validationcurve",dpi=300)
plt.show()

"""
original_airways=airwayvalidationdataold("originalsetupairCA1/logTrain")
original_airways,=plot(original_airways,"original airways")
original_background=backgroundvalidationdataold("originalsetupairCA1/logTrain")
original_background,=plot(original_background,"original background")
plt.axvspan(43, 77, color='red', alpha=0.5,label="15 to 30 epoch cancellation region")
plt.legend(handles=[original_airways,original_background],loc='best')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
plt.show()
"""