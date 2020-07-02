import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow.keras.layers
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pdb
import scipy

#The image file
image='./test_images/orange_dahlia.jpg'
#The model file
saved_model_path = 'saved_model.h5'
#The json  file
jason_file_name = 'label_map.json'
#the top probabilities
k=3



# the resize function 
def process_image(image):
    global dsize
    image_size=224
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image,(image_size, image_size) )
    image/=255
    return image

# the predict function
def predict(image_path=None, model=None, top_k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    print(image_path.split("/")[-1])
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs=model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

with open(jason_file_name, 'r') as f:
    class_names = json.load(f)

    
    
#the json file classes are from 1-102, but the data set labels are from 0-101   
class_names_new = dict()
for key in class_names:
    class_names_new[str(int(key)-1)] = class_names[key]
del class_names
class_names = class_names_new


#Load the training model  
model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer':hub.KerasLayer})

prediction, classes = predict(image_path=image,model=model,top_k=k)


pred=prediction.numpy().squeeze().tolist()
class_pred=classes.numpy().squeeze().tolist()

print(f'\n\n The {image} is a {class_names.get(str(class_pred[int(np.where(np.amax(pred))[0])]))}\n\n')

cont=0
for i in class_pred:
    pred_f=round(pred[cont],2)
    print(f'The probabilty of being {class_names.get(str(i))} is: {pred_f}')
    cont+=1