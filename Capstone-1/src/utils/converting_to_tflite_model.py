import tensorflow as tf
from tensorflow import keras
import os

MODELS_DIRECTORY = '../../models'


def convert_keras_model_to_tflite_model(model_name, output_name):
    """
    Converts TF .h5 model to .tflite model.
    :param model_name: name of a .h5 model
    :param output_name: desired name of .tflite model
    :return: 
    """
    model = keras.models.load_model(os.path.join(MODELS_DIRECTORY, f'{model_name}.h5'))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(os.path.join(MODELS_DIRECTORY, f'{output_name}.tflite'), 'wb') as f_out:
        f_out.write(tflite_model)

    print('Converted')
    print(f'File size {os.path.getsize(os.path.join(MODELS_DIRECTORY,f"{output_name}.tflite")) / (1024 * 1024)}')


if __name__ == '__main__':
    convert_keras_model_to_tflite_model('lemon_rich_enrichments_13_0.998', 'lemon')
