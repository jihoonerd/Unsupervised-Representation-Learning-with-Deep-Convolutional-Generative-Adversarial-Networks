import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Lambda,
    LeakyReLU,
    ReLU,
    Reshape,
)


class Generator(Model):
    def __init__(self, color_image=True):
        super().__init__()
        out_filter = 3 if color_image else 1
        self.dense1 = Dense(
            4 * 4 * 1024,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
            input_shape=(100,),
        )
        self.batch_norm1 = BatchNormalization()
        self.relu1 = ReLU()
        self.pj_and_reshape = Reshape((4, 4, 1024))
        self.conv_tr2 = Conv2DTranspose(
            filters=512, kernel_size=5, strides=2, padding="SAME"
        )
        self.batch_norm2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv_tr3 = Conv2DTranspose(
            filters=256, kernel_size=5, strides=2, padding="SAME"
        )
        self.batch_norm3 = BatchNormalization()
        self.relu3 = ReLU()
        self.conv_tr4 = Conv2DTranspose(
            filters=128, kernel_size=5, strides=2, padding="SAME"
        )
        self.batch_norm4 = BatchNormalization()
        self.relu4 = ReLU()
        self.conv_tr5 = Conv2DTranspose(
            filters=out_filter,
            kernel_size=5,
            strides=2,
            padding="SAME",
            activation="tanh",
        )

    @tf.function
    def __call__(self, input_z, training=False):
        x = self.dense1(input_z)
        x = self.batch_norm1(x, training)
        x = self.relu1(x)
        x = self.pj_and_reshape(x)
        x = self.conv_tr2(x)
        x = self.batch_norm2(x, training)
        x = self.relu2(x)
        x = self.conv_tr3(x)
        x = self.batch_norm3(x, training)
        x = self.relu3(x)
        x = self.conv_tr4(x)
        x = self.batch_norm4(x, training)
        x = self.relu4(x)
        out = self.conv_tr5(x)
        return out


class Discriminator(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME")
        self.lrelu1 = LeakyReLU(0.2)
        self.conv2 = Conv2D(filters=256, kernel_size=5, strides=2, padding="SAME")
        self.batch_norm2 = BatchNormalization()
        self.lrelu2 = LeakyReLU(0.2)
        self.conv3 = Conv2D(filters=512, kernel_size=5, strides=2, padding="SAME")
        self.batch_norm3 = BatchNormalization()
        self.lrelu3 = LeakyReLU(0.2)
        self.conv4 = Conv2D(filters=1, kernel_size=5, strides=2, padding="SAME")
        self.flatten = Flatten()
        self.dense4 = Dense(1)

    @tf.function
    def __call__(self, input_z, training=False):
        x = self.conv1(input_z)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x, training)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        out = self.dense4(x)
        return out
