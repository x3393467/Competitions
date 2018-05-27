def all_path(dirname):  
    #get the files name and the path 
    result = []
    
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
        name = file_name_list
    return result