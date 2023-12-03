import tensorflow as tf
from tensorflow import keras
import click


@click.command()
@click.option(
    "--path_to_model",
    help="Path to h5 model",
    type=str
)
@click.option(
    "--path_to_output_directory",
    help="Path to output directory",
    type=str
)
def convert_to_saved_model_format(path_to_model, path_to_output_directory):
    model = keras.models.load_model(path_to_model)
    tf.saved_model.save(model, path_to_output_directory)


if __name__ == '__main__':
    convert_to_saved_model_format()
