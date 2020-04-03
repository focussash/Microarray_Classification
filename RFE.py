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
        print(features)
    #Finally output selected features
    return features
