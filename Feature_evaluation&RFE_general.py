def feature_evaluation_pretreated(model,features,dataset,labels,k,Nclasses):
    #This function applies K-fold validation to check for the selected features of any given classifier and determines its performance
    #This is specifically designed for datasets pretreated into what sklearn wants, already
    output = []
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
        output = model.predict(final_data_saved[:,features])
        print(final_label_saved)
        #Predictions to be compared against final_label_saved
     
        ######building
        print(output)
    return output

def recursive_feature_elimination_pretreated(model,dataset,labels,Nfeature_total,Nfeature_desired,Nclasses):
    #This function applies RFE with a model until the feature list contains only Nfeatures
    #This is specifically designed for datasets pretreated into what sklearn wants, already

    #Timing modules:
    #start_time = time.time() 
    #print("--- %s seconds ---" % (time.time() - start_time))

    #It selects the features with the largest coefficients in the linear model
    output = []
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

    #Finally output selected features
    return features
