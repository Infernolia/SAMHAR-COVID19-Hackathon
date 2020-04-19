# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:10:21 2020

@author: Ayush Das
"""

## Predicitng Models


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

#model = load_model('model_vgg16.h5')
img = image.load_img('Dataset/val/PNEUMONIA/person1947_bacteria_4876.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = loaded_model.predict(img_data)
