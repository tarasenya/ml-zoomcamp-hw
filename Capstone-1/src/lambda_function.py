import os

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
import numpy as np

from PIL import Image

MODEL_PATH = os.getenv('MODEL_PATH')

interpreter = tflite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(url_to_image):
    img = download_image(url_to_image)
    target_image_size = (300, 300)

    img = prepare_image(img, target_image_size)
    img = np.array(img, dtype='float32')

    img_as_array = np.array([img])
    img_as_array = img_as_array / 255.0

    interpreter.set_tensor(input_index, img_as_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    return prediction[0][0]


def lambda_handler(event, context):
    url_to_image = event['url']
    res = predict(url_to_image)
    print(res)
    if res > 0.5:
        return {'result': 'Good Quality', 'probability': float(res)}
    else:
        return {'result': 'Bad Quality', 'probability': float(1 - res)}
