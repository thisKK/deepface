import os
from pathlib import Path
import gdown
import zipfile

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout

#-------------------------------------

def loadModel(url = 'https://drive.google.com/file/d/1JmEmDO4hODlX2uFJq8dsGISEkhZan49U/view?usp=sharing'):
	base_model = Sequential()
	base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
	base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
	base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
	base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
	base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
	base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
	base_model.add(Flatten(name='F0'))
	base_model.add(Dense(4096, activation='relu', name='F7'))
	base_model.add(Dropout(rate=0.5, name='D0'))
	base_model.add(Dense(8631, activation='softmax', name='F8'))
	
	#---------------------------------
	
	home = str(Path.home())

	output = home + '/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5'
	
	if os.path.isfile(output) != True:
		print("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")
		gdown.download(url, output, quiet=False)

	try:
		base_model.load_weights(output)
	except Exception as err:
		print(str(err))
		print("Pre-trained weight could not be loaded.")
		print("You might try to download the pre-trained weights from the url ", url, " and copy it to the ", output)
	#drop F8 and D0. F7 is the representation layer.
	deepface_model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)
		
	return deepface_model