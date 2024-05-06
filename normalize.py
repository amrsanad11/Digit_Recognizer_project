import tensorflow as tf


def normalize_img(image , label):
    
    return tf.cast(image, tf.float32) / 255.0 , label

