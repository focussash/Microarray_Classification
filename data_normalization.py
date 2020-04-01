import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)
#This code is for data normalization to render the data zero mean and unit variance. Further, this code records the parameters used for normalization
num_capture = 16
num_detection = 16
num_feature_total = num_capture * num_detection
temp = []

paths = glob.glob(os.path.join(os.getcwd(),"processed_data\*"))
temp = []
for file in paths:
    content = pd.read_csv(file,index_col = 0) #Read in the file
    temp.append(content)
fulldata = pd.concat(temp,axis=1, ignore_index=True)  
fulldata = fulldata.replace(0,np.NaN) #Exclude non-measured data from raw data list
normalization_data = np.zeros(shape=(256,4)) #For each feature, 2 means and 2 standard deviations
#0 - 99; 0 - 255
featuretemp = []
for i in range(num_feature_total):
    tempA = []
    tempB = []
    for j in range(len(paths)):
        a = fulldata.iloc[i,(j*20):(j*20+10)]
        b = fulldata.iloc[i,(j*20+10):(j*20+20)]
        tempA.append(a)
        tempB.append(b)
    featuredataA = pd.concat(tempA, ignore_index=True) 
    featuredataB = pd.concat(tempB, ignore_index=True) 

    #Let's save the parameters used for standardization
    normalization_data[i,0] = featuredataA.mean()
    normalization_data[i,2] = featuredataB.mean()
    normalization_data[i,1] = featuredataA.std()
    normalization_data[i,3] = featuredataB.std()

    #Now, standardize the data
    featuredataA = featuredataA.subtract(normalization_data[i,0])
    featuredataA = featuredataA.divide(normalization_data[i,1])
    featuredataB = featuredataB.subtract(normalization_data[i,2])
    featuredataB = featuredataB.divide(normalization_data[i,3])

    #Finally, put normalized data back to a dataset
    featuredataTotal = pd.concat([featuredataA,featuredataB], ignore_index=True) 
    featuretemp.append(featuredataTotal)
processedData = pd.concat(featuretemp, ignore_index=True,axis = 1)
processedData = processedData.T
df_normalization = pd.DataFrame(normalization_data)
#Now, write the processed data to csv files
processedData.to_csv('Standarized.csv',index = False)
df_normalization.to_csv('Normalization_data.csv',index = False)

#Note that now the data structure is slightly different. 
#Out of the 100 rows,first 50 are repeat A (10 samples each class) and second 50 are repeat B (also 10 samples/class)