"""
Module to train a final model using hyperparameters found in models_investigation.ipynb. For preprocessing data we use
pickled preprocessor.
"""
import json
import lightgbm as lgbm
import pandas as pd
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import mean_squared_error

from utils import get_preprocessor


def train_final_model():
    preprocessor = get_preprocessor()

    training_data = pd.read_csv('../data/preprocessed/train_full.csv')
    X_train = training_data.drop(columns=['energy_star_score', 'label_for_target_var'])
    y_train = training_data['energy_star_score']

    testing_data = pd.read_csv('../data/preprocessed/test_full.csv')
    X_test = testing_data.drop(columns=['energy_star_score', 'label_for_target_var'])
    y_test = testing_data['energy_star_score']

    transformed_training_data = preprocessor.transform(X_train)
    transformed_testing_data = preprocessor.transform(X_test)

    training_dataset = lgbm.Dataset(transformed_training_data, label=y_train, free_raw_data=True)
    validation_dataset = lgbm.Dataset(transformed_testing_data,
                                      label=y_test, reference=training_dataset, free_raw_data=True)

    # we use the found parameters found in models_investigation.ipynb
    with open('../model_artifacts/model_parameters.json', 'r') as f_in:
        model_parameters = json.load(f_in)

    lgbm_booster = lgbm.train(model_parameters, train_set=training_dataset, num_boost_round=2000,
                              callbacks=[early_stopping(100), log_evaluation(100)],
                              valid_sets=[training_dataset, validation_dataset])
    lgbm_booster.save_model(r'../model_artifacts/lightbm_booster')
    # verify predictions
    print(lgbm_booster.predict(transformed_training_data))
    print(lgbm_booster.predict(transformed_testing_data))

    print(mean_squared_error(lgbm_booster.predict(transformed_training_data), y_train, squared=False))
    print(mean_squared_error(lgbm_booster.predict(transformed_testing_data), y_test, squared=False))


if __name__ == '__main__':
    train_final_model()
