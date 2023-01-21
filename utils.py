import cv2 
import numpy as np 
from PIL import Image
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing.image import img_to_array
from keras.models import load_model

classifier = load_model('models\\valid_classifier.h5')
localize = load_model('models\\cancer_image_model.h5')

def predict(image_path):
    image = img_to_array(Image.fromarray(cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (128, 128))))
    image /= 255.0
    image = np.expand_dims(image, [0, 3])
    mask_image = localize.predict(image)
    mask_image_path = image_path.split('.')[0] + '_result.png'
    result = classifier.predict(mask_image)
    mask_image = np.squeeze(mask_image)
    mask_image *= 255.0
    cv2.imwrite(mask_image_path, mask_image)
    

    return np.argmax(result[0])
