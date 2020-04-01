import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations
import matplotlib.pyplot as plt
import random

pd.set_option('display.expand_frame_repr', False)
#This code is for raw data processing; i.e. read from the feature_list file to separate raw data into a matrix of
#(class, feature, repeats)

#First load raw data
raw_data = pd.read_csv('Raw_data.csv')
feature_list = pd.read_csv('Feature_list.csv')
capture = feature_list.index #list of capture Abs
detection = feature_list.columns #list of detection Abs
num_capture = 16
num_detection = 16
num_feature_total = num_capture * num_detection

raw_data = raw_data.replace(0,np.NaN) #Exclude non-measured data from raw data list
for i in raw_data.columns: #Each cell line
    data = np.zeros(shape = (num_feature_total,20)) #We have 256 features and maximum 20 samples per feature
    index_current = 0
    for a in range(num_capture):
        for b in range(num_detection):
            #Here we store feature names in terms of numbers to avoid complexities with strings, so there is no more need to store names
            Nsample = feature_list.iloc[b,a+1]
            #Now, here is a bit of trick; since sometimes Nsample is 5 and sometimes 10, we need to adjust the 5 ones
            #Basically, we are doing a psuedo over-sampling by applying RVOS on Nsample = 5 ones so that all of them are N = 10
            if Nsample == 10:
                for c in range(Nsample):
                    data[a*16+b,c] = raw_data.loc[index_current,i]
                    data[a*16+b,c+Nsample] = raw_data.loc[index_current+1600,i]
                    index_current += 1   
            elif Nsample == 5: #Here, we duplicate the data
                index_temp = index_current #Take this as baseline index for RVOS
                for c in range(Nsample):
                    rand_sample = random.sample(range(0,Nsample),1)
                    data[a*16+b,c] = raw_data.loc[index_current,i]
                    data[a*16+b,c+Nsample] = raw_data.loc[index_temp + rand_sample[0],i]
                    data[a*16+b,c+Nsample*2] = raw_data.loc[index_current+1600,i]
                    data[a*16+b,c+Nsample*3] = raw_data.loc[index_temp+1600+ rand_sample[0],i]
                    index_current += 1   
    #Finally, convert to dataframe for writing into CSV
    current_data = pd.DataFrame(data)
    current_data = current_data.replace(0,np.NaN)
    current_data.to_csv(str(i + '.csv'))

#Now, we cast the data for each feature into 0 mean and unit variance
paths = glob.glob(os.path.join(os.getcwd(),"processed_data\*"))
temp = []
for file in paths:
    content = pd.read_csv(file,index_col = 0) #Read in the file
    temp.append(content)
fulldata = pd.concat(temp,axis=1, ignore_index=True)   


#####################################################################
#fig, ax = plt.subplots()  # Create a figure containing a single axes.
#ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
#plt.show()