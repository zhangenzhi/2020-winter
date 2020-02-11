import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

from tensorflow.keras.applications import VGG19

vgg19 = VGG19()
features_list = [layer.output for layer in vgg19.layers]

feat_extraction_model = keras.Model(inputs = vgg19.input,outputs = features_list)

img = np.random.random((1,224,224,3)).astype('float32')
extracted_features = feat_extraction_model(img)