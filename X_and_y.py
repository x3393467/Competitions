def X_and_y(X_path,y_,img_list,ID):
	'''
	split the train data with driver ID 
	'''

    print('按司机ID分类数据')
    X = []
    y = []
    nu = 0

    for i in ID:
        X_array = []
        y_array = []
        for ii in range(len(X_path)):  
            if os.path.basename(X_path[ii]) in img_list.loc[i].values:
                #x_ar = found_features(X_path[ii]) #将在ID下图片的提取特征          
                x_ar = X_path[ii]
                X_array.append(x_ar)    #形式为 [[特征 ],[特征 ] ...]
                y_array.append(y_[ii])  
                print("{}/{} Data named {}.........{}files\r".
                      format(nu+1,len(ID),i,len(img_list.loc[i]))),
                
        X.append(X_array)  #形式为 [ [[ID1_特征 ],[ID1_特征 ] ...] ,[[ID2_特征 ] [ID2_特征 ] ...] ,...]
        y.append(y_array)
        
        nu += 1       
    return X,y