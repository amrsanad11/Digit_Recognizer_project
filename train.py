import os 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras 
import math
from augment import augment
from data import data
from model import my_model
from normalize import normalize_img 
#----------------------------------------------------------------
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0] , True)

#----------------------------------------------------------------
train_path = "train_images"
test_path = "test_images"
# train_ds, test_ds = data(train_path, test_path)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train_images" ,
    labels="inferred",  # Or "categorical" if labels are in separate files
    batch_size=32,
    image_size=(64, 64)  # Adjust image dimensions if needed
)
test_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    "test_images", # Or "categorical" if labels are in separate files
    batch_size=32,
    image_size=(64, 64)  # Adjust image dimensions if needed
)
#----------------------------------------------------------------
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# BATCH_SIZE = 32

# train_ds = train_ds.cache()
# train_ds = train_ds.shuffle(2000).map(normalize_img , num_parallel_calls = AUTOTUNE).map(augment,
# num_parallel_calls = AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# test_ds = test_ds.map(normalize_img, num_parallel_calls = AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

#----------------------------------------------------------------
model  = my_model()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False) ,
    optimizer = keras.optimizers.Adam(learning_rate=1e-4) ,
    metrics = ['accuracy']
)

model.fit(train_ds, epochs=10, batch_size=32)
model.evaluate(test_ds, batch_size=32)
model.save('model')

