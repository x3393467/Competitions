def train_first_model(X_A, X_B, 
                      y_A, y_B, 
                      model_name, 
                      sgd,
                      stack_name,#模型的名称，'_MobileNet_' 或者 '_VGG16_'
                      input_l=224, input_w=224, input_c=3,
                      n_splits=5,
                      epochs=1,
                      flod_number=3
                      ):
    #得到本次训练的 训练集和验证集
    X_data = X_A + X_B
    y_data = y_A + y_B
    
    model = model_name()
    model.compile(optimizer=sgd, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train_model_1(model,
              X_data, y_data,
              input_l=input_l, input_w=input_w, input_c=input_c,
              n_splits=n_splits,
              epochs=epochs,
              flod_number=flod_number,
              stack_name=stack_name)
    return model