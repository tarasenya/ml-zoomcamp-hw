"""
Module that includes create_val_test_directory function to transform kaggle dataset structure to a structure that is
suitable for ImageDataGenerator, dividing the whole dataset into train and validation.
"""
import os
import random
import shutil
import click

random.seed(42)


def copy_to_dir(file_names: list[str], src_folder: str, dst_folder: str):
    """
    Copying file with file_names in src_folder to a dst_folder
    :param file_names: names of files in src_folder
    :param src_folder: source folder
    :param dst_folder: destination folder
    :return:
    """
    for file_name in file_names:
        shutil.copy(os.path.join(src_folder, file_name), dst_folder)


@click.command()
@click.option(
    "--path_to_dataset",
    help="Path to original dataset",
    type=str
)
@click.option(
    "--path_to_output_directory",
    help="Path to output directory",
    type=str
)
def create_val_test_directory(path_to_dataset: str, path_to_output_directory: str):
    """
    Transforms kaggle dataset structure to a format suitable for ImageDataGenerator.
    :param path_to_dataset: path to the top level of kaggle dataset
    :param path_to_output_directory: path to a destination folder with the folders train and val, each of it includes
    two folders good_quality, bad_quality.
    :return:
    """
    n_good_quality = len(os.listdir(os.path.join(path_to_dataset, 'good_quality')))
    n_bad_quality = len(os.listdir(os.path.join(path_to_dataset, 'bad_quality')))

    good_quality_val = random.sample(os.listdir(os.path.join(path_to_dataset, 'good_quality')),
                                     int(n_good_quality * 0.2))

    bad_quality_val = random.sample(os.listdir(os.path.join(path_to_dataset, 'bad_quality')),
                                    int(n_bad_quality * 0.2))

    copy_to_dir(good_quality_val, os.path.join(path_to_dataset, 'good_quality'),
                os.path.join(os.path.join(path_to_output_directory, 'val'), 'good_quality'))

    copy_to_dir(bad_quality_val, os.path.join(path_to_dataset, 'bad_quality'),
                os.path.join(os.path.join(path_to_output_directory, 'val'), 'bad_quality'))

    good_quality_train = list(
        set(os.listdir(os.path.join(path_to_dataset, 'good_quality'))) - set(good_quality_val))

    bad_quality_train = list(
        set(os.listdir(os.path.join(path_to_dataset, 'bad_quality'))) - set(bad_quality_val))

    copy_to_dir(good_quality_train, os.path.join(path_to_dataset, 'good_quality'),
                os.path.join(os.path.join(path_to_output_directory, 'train'), 'good_quality'))

    copy_to_dir(bad_quality_train, os.path.join(path_to_dataset, 'bad_quality'),
                os.path.join(os.path.join(path_to_output_directory, 'train'), 'bad_quality'))


if __name__ == '__main__':
    create_val_test_directory()
