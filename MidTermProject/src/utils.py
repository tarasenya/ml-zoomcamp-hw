"""
Utils module
"""

import lightgbm as lgbm
import pickle

from sklearn.compose import ColumnTransformer


def get_lgbm_booster(model_name: str) -> lgbm.Booster:
    """
    Gets a lgbm booster from a model_artifacts library using Booster class.
    :model_name: name of a saved model.
    :return:
    """
    return lgbm.Booster(model_file=f'../model_artifacts/{model_name}')


def get_preprocessor() -> ColumnTransformer:
    """
    Demarshallizing pickled preprocessing transformer.
    :return:
    """
    with open('../model_artifacts/preprocess.pkl', 'rb') as f_out:
        preprocessor = pickle.load(f_out)
    return preprocessor
