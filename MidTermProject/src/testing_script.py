import requests
import pandas as pd


def send_testing_request(host='localhost:3141') -> dict:
    testing_data = pd.read_csv('../data/preprocessed/test_full.csv').drop(
        columns=['energy_star_score', 'label_for_target_var'])

    response = requests.post(f'http://{host}/predict', json=testing_data.iloc[2].to_dict()).json()
    return response

def call_ping(host='localhost:3141'):
    response = requests.get(f'http://{host}/ping').json()
    return response

if __name__ == '__main__':
    aws_host = 'starscore-serving-env.eba-j3hmc4ay.us-west-2.elasticbeanstalk.com'
    print(send_testing_request(aws_host))
    print(call_ping(aws_host))