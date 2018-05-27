def trainpre(X_data,
                input_l=299,input_w=299, input_c=3):
	'''
	read imgs for train proedict
	'''
    number_of_train_A = 0
    for i in range(len(X_data)):   
        number_of_train_A += len(X_data[i])
                     
    #根据图片数量，创建空矩阵 
    train_feature_A = np.zeros(shape=[number_of_train_A,input_l,input_w, input_c],
                              dtype=np.uint8)
        
    #逐个写入矩阵
    nu_ = 0   
    for i in range(len(X_data)):    
        for N in range(len(X_data[i])):
            train_feature_A[nu_] = img_read(X_data[i][N],input_l,input_w)
            nu_ += 1
            print("{}/{} Ready .........\r".format(nu_+1,number_of_train_A)),
    
    #model = model_name
    #output = model.predict(train_feature_A,verbose=1)
    return train_feature_A

def testpre(img_path,
            input_l=299,input_w=299, input_c=3):
    
    path = all_path(img_path)
    number_of_pre = len(path)
    test_feature = np.zeros(shape=[number_of_pre,input_l,input_w, input_c],
                           dtype=np.uint8)
    for N in range(len(path)):
        test_feature[N] = img_read(path[N],input_l,input_w)
        print("{}/{} Ready .........\r".format(N+1,number_of_pre)),
    return test_feature    