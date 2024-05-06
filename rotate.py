import math
import tensorflow as tf
# import tensorflow_addons as tfa

def rotate(img , max_degrees = 25):
    # degrees = tf.random.uniform([] , -max_degrees , max_degrees , dtype = tf.float32) 
    img = img.rotate(25)
    return img 

