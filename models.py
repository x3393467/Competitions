model_1 = Xception(include_top=False, weights='imagenet',
                        pooling='avg')

model_2 = ResNet50(include_top=False, weights='imagenet',
                       input_shape=(299,299,3), pooling='avg')

model_3 = MobileNet(input_shape=(224,224,3), alpha=1.0, depth_multiplier=1, 
                     dropout=1e-2, include_top=False, 
                     weights='imagenet', input_tensor=None, 
                     pooling='avg')

model_4 = VGG16(include_top=False, weights='imagenet',
                input_tensor=None, input_shape=None,
                pooling='avg')

def preprocess_input(x):
    x /= 255.
    return x

 def MODEL_Xception():
    img_in = Input((299,299,3))
    a = Lambda(preprocess_input,output_shape=(299,299,3))(img_in)
    x = model_1(a)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)  
    model = Model(img_in, x)
    return model

 def MODEL_ResNet50():
    img_in = Input((224,224,3))
    a = Lambda(preprocess_input,output_shape=(224,224,3))(img_in)
    x = model_1(a)
    x = Dropout(0.7)(x)
    x = Dense(10, activation='softmax')(x)  
    model = Model(img_in, x)
    return model

 def MODEL_VGG16():
    img_in = Input((224,224,3))
    a = Lambda(preprocess_input,output_shape=(224,224,3))(img_in)
    x = model_1(a)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)  
    model = Model(img_in, x)
    return model

  def MODEL_MobileNet():
    img_in = Input((224,224,3))
    a = Lambda(preprocess_input,output_shape=(224,224,3))(img_in)
    x = model_1(a)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)  
    model = Model(img_in, x)
    return model