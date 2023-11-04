import json

import pandas as pd
from flask import Flask, request, jsonify
from utils import get_preprocessor, get_lgbm_booster

app = Flask('prediction_app')

MODEL = get_lgbm_booster('lightbm_booster')
PREPROCESSOR = get_preprocessor()


@app.route('/predict', methods=['POST'])
def prediction_end_point():
    preprocessed_request = PREPROCESSOR.transform(pd.DataFrame([request.get_json()]))
    res = {'res': float(MODEL.predict(preprocessed_request)),
           'status': 'OK'}
    return jsonify(res)


@app.route('/ping', methods=['GET'])
def ping():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run(host='localhost', port=3141, debug=True)
