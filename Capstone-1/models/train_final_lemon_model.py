"""
Module for training the final model
"""
import click
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras


def visualize_history(history: tf.keras.callbacks.History, epochs: int):
    """
    Visualizing history of training of NN for the given number of epochs
    :param history: tf.keras.callbacks.History object
    :param epochs: number of epochs to visualize a history
    :return:
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


@click.command()
@click.option(
    "--path_to_lemon_dataset",
    help="Path to original dataset",
    type=str
)
def train_final_lemon_model(path_to_lemon_dataset) -> keras.Sequential:
    """
    Training the model with rich enrichments
    :param path_to_lemon_dataset:  path to lemon dataset (top level with train, val directories, each containing
    good_quality, bad_quality folders).
    :return:
    """
    num_epochs = 20
    learning_rate = 0.001

    train_gen_rich = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.2,
        height_shift_range=0.05,
        vertical_flip=True,
        horizontal_flip=True,
        shear_range=20,
        zoom_range=[0, 1.2],
        rotation_range=30,
    )

    train_ds_rich = train_gen_rich.flow_from_directory(
        os.path.join(path_to_lemon_dataset, 'train'),
        target_size=(300, 300),
        batch_size=32,
    )

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    val_ds = val_gen.flow_from_directory(
        os.path.join(path_to_lemon_dataset, 'val'),
        target_size=(300, 300),
        batch_size=32,
        shuffle=False
    )

    rich_augmentation_model = keras.Sequential([
        keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(300, 300, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    callbacks_rich = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                      keras.callbacks.ModelCheckpoint('lemon_rich_enrichments_{epoch:02d}_{val_accuracy:.3f}.h5',
                                                      save_best_only=True,
                                                      monitor='val_accuracy',
                                                      mode='max')]

    rich_augmentation_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    rich_augmentation_model_history = rich_augmentation_model.fit(train_ds_rich, epochs=num_epochs,
                                                                  validation_data=val_ds,
                                                                  callbacks=callbacks_rich)

    visualize_history(rich_augmentation_model_history, num_epochs)
    return rich_augmentation_model


if __name__ == '__main__':
    train_final_lemon_model()
