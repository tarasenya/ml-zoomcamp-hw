import tensorflow as tf
from tensorflow import keras
import os
def convert_keras_model_to_tflite_model(model_name, output_name):
    model = keras.models.load_model(f'{model_name}.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(f'{output_name}.tflite', 'wb') as f_out:
        f_out.write(tflite_model)

    print('Converted')
    print(f'File size {os.path.getsize(f"{output_name}.tflite")/(1024*1024)}')

if __name__ == '__main__':
    convert_keras_model_to_tflite_model('bees-wasps', 'bee-wasps')