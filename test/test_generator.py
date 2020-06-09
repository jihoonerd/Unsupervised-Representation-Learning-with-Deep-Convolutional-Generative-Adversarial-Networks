import tensorflow as tf

from dcgan.network.networks import Generator


def test_generator():
    tf.config.experimental_run_functions_eagerly(True)

    input_z = tf.random.uniform((1, 100))
    gen = Generator()
    out = gen(input_z)
    assert out.shape == (1, 64, 64, 3)
