import tensorflow as tf
from tensorflow  import keras


def my_model ():
    inputs = keras.layers.Input(shape = (64 , 64 , 1))
    x = keras.layers.Conv2D(32 , 3 )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPooling2D()(x)
    
    x =  keras.layers.Conv2D(64 , 3 )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPooling2D()(x)
    
    x =  keras.layers.Conv2D(128 , 3 )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPooling2D()(x)
    
    x =  keras.layers.Conv2D(64 , 3 )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64 , activation = 'relu')(x)
    outputs = keras.layers.Dense(10  , activation = 'softmax') (x)
    model   = keras.Model(inputs=inputs, outputs=outputs)
    return model
    
    
    