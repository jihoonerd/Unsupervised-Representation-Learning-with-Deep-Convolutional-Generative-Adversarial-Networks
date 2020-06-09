import tensorflow as tf

from dcgan.trainer.trainer import DCGANTrainer


def test_train():
    tf.config.experimental_run_functions_eagerly(True)
    trainer = DCGANTrainer(data="test", color=False, batch_size=64, epochs=1)
    trainer.train()
