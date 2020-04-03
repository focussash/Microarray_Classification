def feature_evaluation(model,dataset,k,Nclasses):
    #This function applies K-fold validation to check for the selected features of any given classifier and determines its performance
    output = []
    data_per_class = 20
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
        random_labels = randnums= np.random.randint(1,Nclasses,final_data_trimmed.shape[0])
        #Now, we can apply data_trimmed to build classifier and data_saved to test classifier

        #model.fit(final_data_trimmed[:,0:3],label_array)
        model.fit(final_data_trimmed,random_labels)
        output = model.predict(final_data_saved)
        ######building
        print(output)
        break
    return output
