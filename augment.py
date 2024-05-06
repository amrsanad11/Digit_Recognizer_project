import tensorflow as tf

from rotate import rotate


def augment(image , label):
    
    image = tf.image.resize(image , size = [64 , 64 ])
    image = rotate (image) 
    image = tf.image.random_brightness(image , max_delta=0.2 )
    image = tf.random_contrast(image , lower = 0.5  , upper = 1.5)
    
    return image , label
