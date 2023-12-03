import os
import grpc
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

from io import BytesIO
from urllib import request as urllib_request
from PIL import Image

INPUTS_INFO = 'conv2d_input'
OUTPUTS_INFO = 'dense_1'

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'lemon-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs[INPUTS_INFO].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = [
    'spoiled',
    'good'
]


def download_image(url):
    with urllib_request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_response(pb_response):
    preds = pb_response.outputs[OUTPUTS_INFO].float_val
    return dict(zip(classes, preds))


def predict(url):
    img = download_image(url)
    target_image_size = (300, 300)

    img = prepare_image(img, target_image_size)
    img = np.array(img, dtype='float32')

    X = np.array([img])
    X = X / 255.0
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
