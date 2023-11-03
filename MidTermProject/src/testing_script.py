import requests
import pandas as pd


def send_testing_request() -> dict:
    testing_data = pd.read_csv('../data/preprocessed/test_full.csv').drop(
        columns=['energy_star_score', 'label_for_target_var'])

    response = requests.post('http://localhost:3141/predict', json=testing_data.iloc[2].to_dict()).json()
    return response


if __name__ == '__main__':
    print(send_testing_request())
