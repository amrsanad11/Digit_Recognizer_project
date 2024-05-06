import tensorflow as tf

saved_dir = 'model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
moedl_lite = converter.convert()

with open ('model.tflite', 'wb') as f :
    f.write(moedl_lite)
    