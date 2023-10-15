from flask import Flask, request, jsonify
import model_utils

app = Flask('prediction_app')

MODEL = model_utils.get_model_pipeline()


@app.route('/predict', methods=['POST'])
def prediction_end_point():
    client_info = request.get_json()
    res = {'res': float(MODEL.predict_proba(client_info)[0][1]), **client_info,
           'gets_credit': str(MODEL.predict(client_info)[0])}
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='localhost', port=3141)
