import tensorflow as tf
import numpy as np
import importlib


def load_data(dataset, keras_dataset=True):
    """
    It only can load datasets from  tf.keras.datasets for now.
    """

    if (
        dataset == "test"
    ):  # TODO: Bad practice here, test case should be handled separately.
        dummy_train_image = np.random.randint(0, 255, size=(64, 64, 64))
        dummy_test_image = np.random.randint(0, 255, size=(64, 64, 64))

        dummy_train_label = np.random.randint(0, 10, size=(64, 1))
        dummy_test_label = np.random.randint(0, 10, size=(64, 1))
        return (
            (dummy_train_image, dummy_test_image),
            (dummy_train_label, dummy_test_label),
        )

    if keras_dataset:
        dataset_lib = importlib.import_module("tensorflow.keras.datasets." + dataset)
        (x_train, y_train), (x_test, y_test) = dataset_lib.load_data()
    else:
        NotImplementedError()
    return (x_train, y_train), (x_test, y_test)


def scale_data(data):
    """
    Assumes input data is uint8 type
    """

    casted = tf.cast(data, tf.float32)
    scaled = (casted - 127.5) / 127.5
    return scaled
