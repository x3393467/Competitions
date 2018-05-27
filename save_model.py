def save_model(model, index, cross=''):
	'''
	create a folder named 'cache'
	save the model_weights in it
	'''
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)