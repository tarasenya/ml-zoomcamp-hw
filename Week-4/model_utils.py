import pickle

from sklearn.pipeline import Pipeline


def get_model_pipeline():
    with open('dv.bin', 'rb') as f_in:
        dict_vectorizer = pickle.load(f_in)

    with open('model2.bin', 'rb') as f_in:
        model = pickle.load(f_in)

    model_pipeline = Pipeline([('dv', dict_vectorizer), ('model', model)])
    return model_pipeline
