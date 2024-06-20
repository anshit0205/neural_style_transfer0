import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential

def VGG16_Avgpool(shape):
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    new_model = Sequential()
    for layer in vgg.layers:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            new_model.add(tf.keras.layers.AveragePooling2D())
        else:
            new_model.add(layer)
    return new_model

def get_content_model(vgg, content_layer_name):
    content_layer = vgg.get_layer(content_layer_name).output
    content_model = Model(vgg.input, content_layer)
    return content_model

def get_style_model(vgg, style_layer_names):
    style_layers = [vgg.get_layer(layer_name).output for layer_name in style_layer_names]
    style_model = Model(vgg.input, style_layers)
    return style_model
