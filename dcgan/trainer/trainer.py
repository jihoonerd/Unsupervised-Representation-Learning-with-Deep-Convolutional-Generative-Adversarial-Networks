import os

import imageio
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from dcgan.data.data_utils import load_data, scale_data
from dcgan.network.networks import Discriminator, Generator


class DCGANTrainer:
    def __init__(
        self,
        data="cifar10",
        keras_dataset=True,
        color=True,
        batch_size=128,
        epochs=300,
        log_interval=1,
    ):
        self.data = data
        self.keras_dataset = True
        self.batch_size = batch_size
        self.epochs = epochs
        self.color_image = color
        self.discriminator = Discriminator()
        self.generator = Generator(color_image=self.color_image)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9)
        self.loss_metric = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.log_interval = log_interval
        self.log_path = (
            "./log/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + self.data
        )
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")

        if not os.path.exists("./outputs"):
            os.mkdir("outputs")
        if not os.path.exists("./log"):
            os.mkdir("log")

    def make_dataset(self, data):

        data = load_data(data, keras_dataset=self.keras_dataset)
        train_images = data[0][0]
        if len(train_images.shape) == 3:
            train_images = train_images[:, :, :, tf.newaxis]
        elif len(train_images.shape) == 4:
            train_images = train_images[:, :, :, :]
        else:
            raise ValueError("Cannot process current data shape")

        scaled_data = scale_data(
            tf.image.resize(images=train_images, size=(64, 64))
        )  # This force images to up/down scale to fit into network suggested in paper.
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(scaled_data)
            .shuffle(scaled_data.shape[0])
            .batch(self.batch_size)
        )
        return train_dataset

    def train(self):
        noise = tf.random.uniform((self.batch_size, 100))  # For test
        train_dataset = self.make_dataset(self.data)

        for epoch in tqdm(range(self.epochs)):
            for minibatch in train_dataset:
                gen_loss, disc_loss = self.train_step(minibatch)
                print(
                    "Epoch: {} => generator loss: {} | discriminator loss: {}".format(
                        epoch, gen_loss, disc_loss
                    )
                )

            if epoch % self.log_interval == 0:
                self.export_generated_image(epoch, noise)
            self.write_summary(epoch, gen_loss, disc_loss)

    @tf.function
    def train_step(self, images):
        noise = tf.random.uniform((self.batch_size, 100))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_images = self.generator(noise, training=True)
            real_out = self.discriminator(images, training=True)
            fake_out = self.discriminator(gen_images, training=True)

            gen_loss = self.calc_generator_loss(fake_out)
            disc_loss = self.calc_discriminator_loss(real_out, fake_out)

        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_grad, self.generator.trainable_variables)
        )

        disc_grad = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grad, self.discriminator.trainable_variables)
        )
        return gen_loss, disc_loss

    @tf.function
    def calc_discriminator_loss(self, real_out, fake_out):
        real_loss = self.loss_metric(tf.ones_like(real_out), real_out)
        fake_loss = self.loss_metric(tf.zeros_like(fake_out), fake_out)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def calc_generator_loss(self, fake_out):
        return self.loss_metric(tf.ones_like(fake_out), fake_out)

    def export_generated_image(self, epoch, test_input):
        outputs = self.generator(test_input)
        fig_size = 3
        plt.figure(figsize=(fig_size, fig_size))
        for i in range(fig_size ** 2):
            plt.subplot(fig_size, fig_size, i + 1)
            if self.color_image:
                outputs_rect = outputs[i, :, :, :]
                cmap = None
            else:
                outputs_rect = outputs[i, :, :, 0]
                cmap = "gray"
            plt.imshow(tf.cast(outputs_rect * 127.5 + 127.5, tf.uint8), cmap=cmap)
            plt.axis("off")
        plt.savefig("./outputs/images_at_epoch_{:04d}.png".format(epoch))

    def write_summary(self, epoch, gen_loss, disc_loss):

        with self.summary_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=epoch + 1)
            tf.summary.scalar("Discriminator Loss", disc_loss, step=epoch + 1)
