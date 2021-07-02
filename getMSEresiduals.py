import numpy as np

meanLoss = np.loadtxt("meanLoss.dat", usecols=[1, -1])
currEpoch = meanLoss[:,0].max()
#mse = np.zeros(int(currEpoch))
for i in range(1,int(currEpoch)):
    print( meanLoss[np.where(meanLoss[:,0]==i),1].mean() )

