def img_load(path):
    #load img path,load targets from folder
    data = load_files(path)
    files = data['filenames']
    #targets = np_utils.to_categorical(np.array(data['target']), 10)
    targets = data['target']

    return files, targets  