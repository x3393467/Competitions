ID,classname,list_img_name,img_list = driver_id('driver_imgs_list.csv',index_col = 'subject')

X_path,y_ = img_load('train')
len(X_data)

#训练集分成3个部分 
X_1 = X_data[:9]
y_1 = y_data[:9] 

X_2 = X_data[9:18]
y_2 = y_data[9:18]

X_3 = X_data[18:]
y_3 = y_data[18:]

pred_train_A = trainpre(X_3, input_l=224,input_w=224, input_c=3)
pred_train_B = trainpre(X_1, input_l=224,input_w=224, input_c=3)
pred_train_C = trainpre(X_2, input_l=224,input_w=224, input_c=3)

#准备先预测一次的test集
predate_test =testpre('test',
            input_l=224,input_w=224, input_c=3)


########################
#train model 1step
#######################
sgd_1 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06
	#训练X_1, X_2, 预测 X_3
model_M_A = train_first_model(X_1, X_2, 
                      y_1, y_2, 
                      model_name=MODEL_MobileNet, 
                      sgd=sgd_1,
                      stack_name='_MobileNet_01',#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=5,
                      flod_number=2
                      )
model_M_A.load_weights('logs/best_weights.df5')
#model_M_A = MODEL_MobileNet()
#model_M_A.load_weights('cache/model_weights3_MobileNet_01.h5')
#预测 X_3
pred_M_train_A = model_M_A.predict(pred_train_A,verbose=1)
np.save('pred_files/pred_M_train_A.npy',pred_M_train_A)
#预测 test
pred_M_test_A = model_M_A.predict(predate_test,verbose=1)
np.save('pred_files/pred_M_test_A.npy',pred_M_test_A)
#如果实例自动关闭 先载入
pred_M_train_A = np.load('pred_files/pred_M_train_A.npy')
pred_M_test_A = np.load('pred_files/pred_M_test_A.npy')

########################################################
# ## 第一个模型
# ### 1-2次训练
sgd_2 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#训练X_2, X_3, 预测 X_1
model_M_B = train_first_model(X_2, X_3, 
                      y_2, y_3, 
                      model_name=MODEL_MobileNet, 
                      sgd=sgd_2,
                      stack_name='_MobileNet_B',#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=1,
                      flod_number=3
                      )
#如果实例自动关闭 先载入
#model_M_B = MODEL_MobileNet()
#model_M_B.load_weights('cache/model_weights3_MobileNet_B.h5')

pred_M_train_B = model_M_B.predict(pred_train_B,verbose=1)
np.save('pred_files/pred_M_train_B.npy',pred_M_train_B)

#预测 test
pred_M_test_B = model_M_B.predict(predate_test,verbose=1)
np.save('pred_files/pred_M_test_B.npy',pred_M_test_B)

#如果实例自动关闭 先载入
#pred_M_train_B = np.load('pred_files/pred_M_train_B.npy')
#pred_M_test_B = np.load('pred_files/pred_M_test_B.npy')

########################################
# ## 第一个模型
# ### 1-3次训练
sgd_3 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#训练X_1, X_3, 预测 X_2
model_M_C = train_first_model(X_1, X_3, 
                      y_1, y_3, 
                      model_name=MODEL_MobileNet, 
                      sgd=sgd_3,
                      stack_name='_MobileNet_C',#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=1,
                      flod_number=3
                      )
#如果实例自动关闭 先载入
#model_M_C = MODEL_MobileNet()
#model_M_C.load_weights('cache/model_weights3_MobileNet_C.h5')

#预测 X_2
pred_M_train_C =  model_M_C.predict(pred_train_C,verbose=1)
np.save('pred_files/pred_M_train_C.npy',pred_M_train_C)

#预测 test
pred_M_test_C = model_M_C.predict(predate_test,verbose=1)
np.save('pred_files/pred_M_test_C.npy',pred_M_test_C)

#如果实例自动关闭 先载入
#pred_M_train_C = np.load('pred_files/pred_M_train_C.npy')
#pred_M_test_C = np.load('pred_files/pred_M_test_C.npy')

#########################################
# # 以下开始第二个模型的3次训练
# ## 第二个模型
# ### 2-1训练
sgd_2_1 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#训练X_1, X_2, 预测 X_3
model_V_A = train_first_model(X_1, X_2, 
                      y_1, y_2, 
                      model_name=MODEL_VGG16, 
                      sgd=sgd_2_1,
                      stack_name='_VGG16_A',#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=1,
                      flod_number=3
                      )
#如果实例自动关闭 先载入
#model_V_A = MODEL_VGG16()
#model_V_A.load_weights('cache/model_weights3_VGG16_A.h5')
#训练X_1, X_2, 预测 X_3
pred_V_train_A = model_V_A.predict(pred_train_A,verbose=1)
np.save('pred_files/pred_V_train_A.npy',pred_V_train_A)
#预测test
pred_V_test_A = model_V_A.predict(predate_test,verbose=1)
np.save('pred_files/pred_V_test_A.npy',pred_V_test_A)
#如果实例自动关闭 先载入
#pred_V_train_A = np.load('pred_files/pred_V_train_A.npy')
#pred_V_test_A = np.load('pred_files/pred_V_test_A.npy')

###############################
# ## 第二个模型
# ### 2-2训练
sgd_2_2 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#训练X_2, X_3, 预测 X_1
model_V_B = train_first_model(X_2, X_3, 
                      y_2, y_3, 
                      model_name=MODEL_VGG16, 
                      sgd=sgd_2_2,
                      stack_name='_VGG16_B',#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=1,
                      flod_number=3
                      )
#如果实例自动关闭 先载入
#model_V_B = MODEL_VGG16()
#model_V_B.load_weights('cache/model_weights3_VGG16_B.h5')
#训练X_2, X_3, 预测 X_1
pred_V_train_B = model_V_B.predict(pred_train_B,verbose=1)
np.save('pred_files/pred_V_train_B.npy',pred_V_train_B)

pred_V_test_B = model_V_B.predict(predate_test,verbose=1)
np.save('pred_files/pred_V_test_B.npy',pred_V_test_B)
#如果实例自动关闭 先载入
#pred_V_train_B = np.load('pred_files/pred_V_train_B.npy')
#pred_V_test_B = np.load('pred_files/pred_V_test_B.npy')

#########################
# ## 第二个模型
# ### 2-3训练
sgd_2_3 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#训练X_1, X_3, 预测 X_2
model_V_C = train_first_model(X_1, X_3, 
                      y_1, y_3, 
                      model_name=MODEL_VGG16, 
                      sgd=sgd_2_3,
                      stack_name='_VGG16_C',#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=1,
                      flod_number=3
                      )
#如果实例自动关闭 先载入
#model_V_C = MODEL_VGG16()
#model_V_C.load_weights('cache/model_weights3_VGG16_C.h5')
#训练X_1, X_3, 预测 X_2
pred_V_train_C = model_V_C.predict(pred_train_C,verbose=1)
np.save('pred_files/pred_V_train_C.npy',pred_V_train_C)

#预测test
pred_V_test_C = model_V_C.predict(predate_test,verbose=1)
np.save('pred_files/pred_V_test_C.npy',pred_V_test_C)
#如果实例自动关闭 先载入
#pred_V_train_C = np.load('pred_files/pred_V_train_C.npy')
#pred_V_test_C = np.load('pred_files/pred_V_test_C.npy')
