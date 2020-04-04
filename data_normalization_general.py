import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations
import matplotlib.pyplot as plt
import random

pd.set_option('display.expand_frame_repr', False)
#This code is for data normalization to render the data zero mean and unit variance. 
#This code is designed for datasets already organized in n_samples X n_features, and labels already in n_samples format
temp = []
df_data = pd.read_csv('Online_data.csv')
df_labels = pd.read_csv('Online_label.csv')
data = df_data.to_numpy()
labels = df_labels.to_numpy()
labels = labels.flatten()

Nclasses = len(np.unique(labels))
#Nsamples_class is automatically set to the class with max samples
Nsamples_class = max(np.bincount(labels))
Nfeatures = data.shape[1]

featuretemp = []
#First, balance all the classes by RVOS, if they were not already balanced
data_balanced = data

for i in np.unique(labels):
    while np.count_nonzero(labels == i) < Nsamples_class:
        index_range = np.where(labels == i)[0].tolist()
        rand_index = random.sample(index_range,1)
        rand_data = data[rand_index,:]
        data_balanced = np.insert(data_balanced,np.count_nonzero(labels == i),rand_data,axis = 0)
        labels = np.insert(labels,np.count_nonzero(labels == i),i)
        print('Inserted a data for label', i)
        print(Nsamples_class - np.count_nonzero(labels == i),'more to go')

#Data_balanced is now ready to be standarized!

mean = data_balanced.mean(axis = 0)
variance = data_balanced.std(axis = 0)
data_balanced = np.subtract(data_balanced,mean)
data_balanced = np.divide(data_balanced,variance)
normalization_data = np.concatenate((mean,variance),axis = 1)  #These are the data used for normalization; for each feature, a mean and a standard deviations

#Finally, write to csv
processedData = pd.DataFrame(data_balanced)
df_normalization = pd.DataFrame(normalization_data)
labels_balanced = pd.DataFrame(labels)
processedData.to_csv('Standarized_online.csv',index = False)
df_normalization.to_csv('Normalization_data_online.csv',index = False)
labels_balanced.to_csv('labels_balanced_online.csv',index = False)
#Note now all three files have header; that isnt a problem, normal read_csv will remove it