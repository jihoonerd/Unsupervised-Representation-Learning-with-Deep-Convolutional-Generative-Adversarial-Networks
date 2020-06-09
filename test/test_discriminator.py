import tensorflow as tf

from dcgan.network.networks import Discriminator, Generator


def test_disgriminator():
    tf.config.experimental_run_functions_eagerly(True)

    input_z = tf.random.uniform((1, 100))
    gen = Generator()
    out_gen = gen(input_z)

    discriminator = Discriminator()
    out_disc = discriminator(out_gen)
    assert out_disc
