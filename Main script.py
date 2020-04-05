import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import accuracy_score,matthews_corrcoef
import sys
import time

pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(threshold=sys.maxsize)
#This is the code containing all relevant functions excluding data preprocessing and normalization

#This part, the data loading, is specific to which dataset to use

##For our lab-generated data
#df = pd.read_csv('Standarized.csv')
#dn = df.to_numpy()

#For online dataset
df = pd.read_csv('Standarized_online.csv')
dn = df.to_numpy()
dl = pd.read_csv('labels_balanced_online.csv')
dln = dl.to_numpy()

#For our dataset
def feature_evaluation(model,features,dataset,k,Nclasses):
    #This function applies K-fold validation to check for the selected features of any given classifier and determines its performance
    output = []
    data_per_class = 20
    acc=[]
    mcc=[]
    for iterations in range(k):
        data_saved = []
        data_trimmed = []
        data_total = []
        for i in range(Nclasses):
            #First subdivide dataset into each class
            current_dataA = dataset[:,(i*10):(i*10+10)] #We have 10 data per class for repeat A after preprocessing
            current_dataB = dataset[:,(i*10+50):(i*10+60)] #We have 10 data per class for repeat B after preprocessing
            current_data = np.concatenate((current_dataA,current_dataB),axis = 1) 
            #Now we have all data for 1 class
            #Now do sampling and remove sampled data from trainingset, to be used as test set
            samples = random.sample(range(0,data_per_class),data_per_class//k)
            saved = current_data[:,samples]
            trimmed = np.delete(current_data,samples,1) #These are data kept for testing 
            data_saved.append(saved)
            data_trimmed.append(trimmed)
        #Next, cast the data to what sklearn likes for both training and testing, in (n_samples,n_features) and construct the label array for sklearn
        label_array = np.zeros(Nclasses*(data_per_class - data_per_class//k))
        true_label_array = np.zeros(Nclasses*(data_per_class//k)) #For saving labels for test data
        for i in range(Nclasses):
            temp_data = np.array(data_trimmed[i])
            temp_saved = np.array(data_saved[i])
            if i > 0:
               final_data = np.concatenate((final_data,temp_data),axis = 1)
               final_data_saved = np.concatenate((final_data_saved,temp_saved),axis = 1)
            elif i == 0:
                final_data = temp_data
                final_data_saved = temp_saved
            for j in range(data_per_class - data_per_class//k): #For generating array of labels for training
                label_array[i*(data_per_class - data_per_class//k)+j] = i
            for j in range(data_per_class//k):
                true_label_array[i*data_per_class//k + j] = i

        final_data_trimmed = final_data.T #In this case, final_data_trimmed is 80X256 for k = 5
        final_data_saved = final_data_saved.T #And the saved testing set is 20X256 for k = 5
        random_labels =  np.random.randint(1,Nclasses,final_data_trimmed.shape[0])
        #Now, we can apply data_trimmed to build classifier and data_saved to test classifier

        model.fit(final_data_trimmed[:,features],label_array)
        #model.fit(final_data_trimmed[:,features],random_labels)
        prediction = model.predict(final_data_saved[:,features])

        ##Now,get prediction accuracy and Matthews correlation coefficient
        acc.append(accuracy_score(prediction,true_label_array))
        mcc.append(matthews_corrcoef(prediction,true_label_array))
    #Now, get mean and average of performance
    output.append(np.array(acc).mean())
    output.append(np.array(acc).std())
    output.append(np.array(mcc).mean())
    output.append(np.array(mcc).std())
    #Final outout is an array of 4 elements [mean Acc, std Acc, mean Mcc, std Mcc]
    print(output)
    return output

def recursive_feature_elimination(model,dataset,Nfeature_total,Nfeature_desired,Nclasses):
    #This function applies RFE with a model until the feature list contains only Nfeatures
    #It selects the features with the largest coefficients in the linear model
    output = []
    data_per_class = 20
    features = np.arange(Nfeature_total) 
    cut_initial = Nfeature_total//10
    cut = cut_initial #This is the amount of features to remove each run of RFE

    #First, build the data for training
    for i in range(Nclasses):

        #First subdivide dataset into each class
        current_dataA = dataset[:,(i*10):(i*10+10)] #We have 10 data per class for repeat A after preprocessing
        current_dataB = dataset[:,(i*10+50):(i*10+60)] #We have 10 data per class for repeat B after preprocessing
        current_data = np.concatenate((current_dataA,current_dataB),axis = 1) 
        #Now we have all data for 1 class
        #Here we dont use sampling; we will use all data to build feature selector
        if i == 0:
            data_total = current_data
        elif i >0:
            data_total = np.concatenate((data_total,current_data),axis = 1)
    #Next, cast the data to what sklearn likes for both training and testing, in (n_samples,n_features) and construct the label array for sklearn
    label_array = np.zeros(Nclasses*data_per_class)
    for i in range(Nclasses):
        for j in range(data_per_class): #For generating array of labels for training
            label_array[i*data_per_class+j] = i

    data_total = data_total.T 

    #Now apply RFE
    temp_length = len(features)
    while len(features)>Nfeature_desired: 
        current_cutoff = len(features) - cut
        #Now train the model and get the coefficients
        model.fit(data_total[:,features],label_array)

        #We find the average of coefficients for a particular feature, because svm builds classifiers for 1VS1 class
        coeffs = abs(clf.coef_.mean(axis = 0)) 
        coeffs_partitioned = np.argpartition(-coeffs,current_cutoff) #This gives largest current_cutoff elements of the coefficients
        coeffs_partitioned = coeffs_partitioned[:current_cutoff]
        features = features[coeffs_partitioned] #Eliminates features with too small coefficients

        #Update variables for RFE
        if len(features)<=(temp_length//2):
            temp_length = len(features)
            if cut > 1:
                cut = cut//2
        #At this point, we can evaluate the model performance with current remaining features, if we want
        #feature_scores = feature_evaluation(clf,features,dataset,5,Nclasses)
    #Finally output selected features
    return features

def one_feature_elimination(model,dataset,Nfeature_total,Nfeature_desired,Nclasses):
    #This function applies feature elimination one-by-one with a model until the feature list contains only Nfeatures

    #It selects the features with the largest coefficients in the linear model
    output = []
    data_per_class = 20
    features = np.arange(Nfeature_total) 
    cut = 1

    #First, build the data for training
    for i in range(Nclasses):

        #First subdivide dataset into each class
        current_dataA = dataset[:,(i*10):(i*10+10)] #We have 10 data per class for repeat A after preprocessing
        current_dataB = dataset[:,(i*10+50):(i*10+60)] #We have 10 data per class for repeat B after preprocessing
        current_data = np.concatenate((current_dataA,current_dataB),axis = 1) 
        #Now we have all data for 1 class
        #Here we dont use sampling; we will use all data to build feature selector
        if i == 0:
            data_total = current_data
        elif i >0:
            data_total = np.concatenate((data_total,current_data),axis = 1)
    #Next, cast the data to what sklearn likes for both training and testing, in (n_samples,n_features) and construct the label array for sklearn
    label_array = np.zeros(Nclasses*data_per_class)
    for i in range(Nclasses):
        for j in range(data_per_class): #For generating array of labels for training
            label_array[i*data_per_class+j] = i

    data_total = data_total.T 

    #Now apply RFE
    temp_length = len(features)
    while len(features)>Nfeature_desired: 
        current_cutoff = len(features) - cut
        #Now train the model and get the coefficients
        model.fit(data_total[:,features],label_array)

        #We find the average of coefficients for a particular feature, because svm builds classifiers for 1VS1 class
        coeffs = abs(clf.coef_.mean(axis = 0)) 
        coeffs_partitioned = np.argpartition(-coeffs,current_cutoff) #This gives largest current_cutoff elements of the coefficients
        coeffs_partitioned = coeffs_partitioned[:current_cutoff]
        features = features[coeffs_partitioned] #Eliminates features with too small coefficients
        #At this point, we can evaluate the model performance with current remaining features, if we want
        #feature_scores = feature_evaluation(clf,features,dataset,5,Nclasses)
    #Finally output selected features
    return features

#For online dataset
def feature_evaluation_pretreated(model,features,dataset,labels,k,Nclasses):
    #This function applies K-fold validation to check for the selected features of any given classifier and determines its performance
    #This is specifically designed for datasets pretreated into what sklearn wants, already
    output = []
    acc = []
    mcc = []
    data_per_class = 51
    for iterations in range(k):
        data_saved = []
        data_trimmed = []
        label_saved = []
        label_trimmed = []
        data_total = []
        for i in range(Nclasses):
            #First subdivide dataset into each class
            current_data = dataset[i*data_per_class:i*data_per_class+data_per_class,:]
            current_label = labels[i*data_per_class:i*data_per_class+data_per_class]
            #Now we have all data for 1 class
            #Now do sampling and remove sampled data from trainingset, to be used as test set
            samples = random.sample(range(0,data_per_class),data_per_class//k)
            saved = current_data[samples,:]
            saved_label = current_label[samples]           
            trimmed = np.delete(current_data,samples,0) #These are data kept for testing 
            trimmed_label = np.delete(current_label,samples)
            data_saved.append(saved)
            label_saved.append(saved_label)
            data_trimmed.append(trimmed)
            label_trimmed.append(trimmed_label)
        #Next, cast the data to what sklearn likes for both training and testing, in (n_samples,n_features) 

        for i in range(Nclasses):
            temp_data = np.array(data_trimmed[i])
            temp_label = np.array(label_trimmed[i])
            temp_saved = np.array(data_saved[i])
            temp_label_saved = np.array(label_saved[i])
            if i > 0:
               final_data = np.concatenate((final_data,temp_data))
               final_label = np.concatenate((final_label,temp_label))
               final_data_saved = np.concatenate((final_data_saved,temp_saved))
               final_label_saved = np.concatenate((final_label_saved,temp_label_saved))
            elif i == 0:
                final_data = temp_data
                final_label = temp_label
                final_data_saved = temp_saved
                final_label_saved = temp_label_saved

        
        random_labels = randnums= np.random.randint(1,Nclasses,final_data.shape[0])
        final_label_saved = final_label_saved.flatten()
        final_label = final_label.flatten()
        #Now, we can apply data_trimmed to build classifier and data_saved to test classifier
        model.fit(final_data[:,features],final_label)
        #model.fit(final_data,random_labels)
        prediction = model.predict(final_data_saved[:,features])
        ##Now,get prediction accuracy and Matthews correlation coefficient
        acc.append(accuracy_score(prediction, final_label_saved))
        mcc.append(matthews_corrcoef(prediction, final_label_saved))
    #Now, get mean and average of performance
    output.append(np.array(acc).mean())
    output.append(np.array(acc).std())
    output.append(np.array(mcc).mean())
    output.append(np.array(mcc).std())
    #Final outout is an array of 4 elements [mean Acc, std Acc, mean Mcc, std Mcc]
    print(output)       
    return output

def recursive_feature_elimination_pretreated(model,dataset,labels,Nfeature_total,Nfeature_desired,Nclasses):
    #This function applies RFE with a model until the feature list contains only Nfeatures
    #It selects the features with the largest coefficients in the linear model
    #This is specifically designed for datasets pretreated into what sklearn wants, already

    output = []
    labels = labels.flatten()
    data_per_class = 51 #Amount of data per class for the online set
    features = np.arange(Nfeature_total) 
    cut_initial = Nfeature_total//10
    cut = cut_initial #This is the amount of features to remove each run of RFE

    #The online data are readily in trainning data and labels format

    #Now apply RFE
    temp_length = len(features)
    while len(features)>Nfeature_desired: 
        current_cutoff = len(features) - cut
        #Now train the model and get the coefficients
        model.fit(dataset[:,features],labels)

        #We find the average of coefficients for a particular feature, because svm builds classifiers for 1VS1 class
        coeffs = abs(clf.coef_.mean(axis = 0)) 
        coeffs_partitioned = np.argpartition(-coeffs,current_cutoff) #This gives largest current_cutoff elements of the coefficients
        coeffs_partitioned = coeffs_partitioned[:current_cutoff]
        features = features[coeffs_partitioned] #Eliminates features with too small coefficients

        #Update variables for RFE
        if len(features)<=(temp_length//2):
            temp_length = len(features)
            if cut > 1:
                cut = cut//2

        #At this point, we can evaluate the model performance with current remaining features, if we want
        #feature_scores = feature_evaluation(clf,features,dataset,5,Nclasses)
        print(len(features))
    #Finally output selected features
    return features

def one_feature_elimination_pretreated(model,dataset,labels,Nfeature_total,Nfeature_desired,Nclasses):
    #This function applies feature elimination one at a time with a model until the feature list contains only Nfeatures
    #It selects the features with the largest coefficients in the linear model
    #This is specifically designed for datasets pretreated into what sklearn wants, already

    output = []
    labels = labels.flatten()
    data_per_class = 51 #Amount of data per class for the online set
    features = np.arange(Nfeature_total) 
    cut = 1

    #The online data are readily in trainning data and labels format

    temp_length = len(features)
    while len(features)>Nfeature_desired: 
        current_cutoff = len(features) - cut
        #Now train the model and get the coefficients
        model.fit(dataset[:,features],labels)

        #We find the average of coefficients for a particular feature, because svm builds classifiers for 1VS1 class
        coeffs = abs(clf.coef_.mean(axis = 0)) 
        coeffs_partitioned = np.argpartition(-coeffs,current_cutoff) #This gives largest current_cutoff elements of the coefficients
        coeffs_partitioned = coeffs_partitioned[:current_cutoff]
        features = features[coeffs_partitioned] #Eliminates features with too small coefficients

        #At this point, we can evaluate the model performance with current remaining features, if we want
        #feature_scores = feature_evaluation(clf,features,dataset,5,Nclasses)
    #Finally output selected features
    return features


#clf = svm.SVC(kernel='linear', C=1.0, probability=True)   # use SVM
clf = svm.LinearSVC(penalty='l1',loss='squared_hinge',C=0.7, dual=False,fit_intercept=True,random_state=1,max_iter = 1000000)   # use LLSVM
Nfeature_total = 256
Nfeature_desired = 5

random_features = random.sample(range(0,Nfeature_total),Nfeature_desired)

start_time = time.time() 
featureset = np.arange(256)
A = feature_evaluation(clf,featureset,dn,5,5)
print("--- %s seconds ---" % (time.time() - start_time))

#cutoffs = [1,5,20,50,100]
#for i in cutoffs:
#    Nfeature_desired = i
#    start_time = time.time() 
#    #featureset = recursive_feature_elimination(clf,dn,Nfeature_total,Nfeature_desired,5)
#    #featureset = random.sample(range(0,Nfeature_total),Nfeature_desired)
#    featureset = np.arange(256)
#    A = feature_evaluation(clf,featureset,dn,5,5)
#    print("--- %s seconds ---" % (time.time() - start_time))

#clf.fit(X,Y)
#print(A)

#####################################################################
#fig, ax = plt.subplots()  # Create a figure containing a single axes.
#ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
#plt.show()

