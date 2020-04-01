def cross_validation(classifier_type,dataset,k,Nclasses):
    #This function applies K-fold validation to any given classifier and determines its performance
    for iterations in range(k):
        data_saved = []
        data_trimmed = []
        data_total = []
        for i in range(Nclasses):
            #First subdivide dataset into each class
            current_dataA = dataset[:,(i*10):(i*10+10)] #We have 10 data per class for repeat A after preprocessing
            current_dataB = dataset[:,(i*10+50):(i*10+60)] #We have 10 data per class for repeat B after preprocessing
            current_data = np.concatenate((current_dataA,current_dataB),axis = 1)
            #Now do sampling and remove sampled data from trainingset, to be used as test set
            samples = random.sample(range(0,20),20//k)
            saved = current_data[:,samples]
            trimmed = np.delete(current_data,samples,1) #These are data kept for testing 
            data_saved.append(saved)
            data_trimmed.append(trimmed)
            print(data_trimmed)
        #Now, we can apply data_trimmed to build classifier and data_saved to test classifier

        ######building
        break
    return data_total
