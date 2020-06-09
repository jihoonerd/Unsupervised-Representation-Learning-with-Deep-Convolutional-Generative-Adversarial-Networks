import tensorflow as tf

from dcgan.data.data_utils import load_data, scale_data


def test_load_data():

    (x_train, y_train), (x_test, y_test) = load_data("cifar10")

    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000, 1)


def test_scale_data():

    (x_train, _), (x_test, _) = load_data("cifar10")

    train_scaled = scale_data(x_train)
    test_scaled = scale_data(x_test)

    assert tf.reduce_min(train_scaled) >= -1.0
    assert tf.reduce_max(train_scaled) <= 1.0
    assert tf.reduce_min(test_scaled) >= -1.0
    assert tf.reduce_max(test_scaled) <= 1.0
