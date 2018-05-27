def img_read(path,img_cols=299, img_rows=299):
    img = cv2.imread(path).astype(np.uint8)
    img_newsize = cv2.resize(img,(img_cols,img_rows))
    return img_newsize #np.expand_dims(img_newsize, axis=2)