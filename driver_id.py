def driver_id(path,index_col = None):
    '''
    fined the ID form CSV
    there'r 26 drivers in this case
    '''
    img_list = pd.read_csv(path,index_col=index_col) #指定索引读取CSV文件
    
    #
    id = img_list.index
    id = set(id) #去重复
    id = list(id)#输出为没有重复ID的列表
    classname = img_list.classname
    list_img_name = img_list.img
    return id,classname,list_img_name,img_list