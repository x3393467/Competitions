def train_model_1(model,X_data_A, y_data_A,
                  input_l=299, input_w=299, input_c=3,
                  n_splits=5,
                  epochs=1,
                  flod_number=1,
                  stack_name='_stackA_'):
'''
train first model
use KFold
'''
    kf = KFold(n_splits=n_splits, shuffle=True)
    n=0
    for train_index_A,verify_index_A in kf.split(X_data_A): 
    
    #计算训练集、验证集 内特征数目   
        number_of_train_A = 0
        number_of_verify_A = 0
        for i in train_index_A:   
            number_of_train_A += len(X_data_A[i])
        for e in verify_index_A:
            number_of_verify_A += len(X_data_A[e])  
        print([number_of_train_A,input_l,input_w, input_c])
             
    #根据图片数量，创建空矩阵 
        train_feature_A = np.zeros(shape=[number_of_train_A,input_l,input_w, input_c],
                                   dtype=np.uint8)
        train_label_A = np.zeros(shape=[number_of_train_A, 10])

        verify_feature_A = np.zeros(shape=[number_of_verify_A,input_l,input_w, input_c],
                                   dtype=np.uint8)
        verify_label_A = np.zeros(shape=[number_of_verify_A, 10])
    
    #逐个写入矩阵
        nu_ = 0
        print('train idex is {},verify index is{}.'.format(train_index_A,verify_index_A))
        for i in train_index_A:    
        
            for N in range(len(X_data_A[i])):
                train_feature_A[nu_] = img_read(X_data_A[i][N],input_l, input_w)  
                train_label_A[nu_] = (np_utils.to_categorical(np.array(y_data_A[i][N]),10))  
                
                nu_ += 1
          
        nu2_ = 0    
        for ii in verify_index_A:   
            for N in range(len(X_data_A[ii])):
                verify_feature_A[nu2_] = img_read(X_data_A[ii][N],input_l, input_w)          
                verify_label_A[nu2_] = (np_utils.to_categorical(np.array(y_data_A[ii][N]),10))
            
                nu2_ += 1
    
        print('X_data is ready')
       checkpointer = ModelCheckpoint(filepath='logs/best_weights.df5',
                        monitor='val_loss',
                        save_best_only=True, 
                        verbose=0, 
                        mode='auto',
                        period=0)
        
        
        
        model.fit(train_feature_A, train_label_A,
                  validation_data=(verify_feature_A,verify_label_A),
                 batch_size = 64,
                          callbacks = [checkpointer],
                          epochs=epochs,
                          verbose=1, 
                          shuffle=True)
       
        n += 1
        if n == flod_number:
            save_model(model ,n,stack_name)
            return model
            break
        save_model(model ,n,stack_name)


