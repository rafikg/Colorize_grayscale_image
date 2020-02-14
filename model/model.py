from typing import Callable

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.regularizers import l2


class ColorizeGrayScale(Model):
    "Encapsulate the encoder-decoder network"

    def __init__(self, is_training: bool, l2_reg: Callable,
                 name='Colorize_grayscale',
                 in_h: int = 224, in_w: int = 224):
        super().__init__(name=name)
        self.is_training = is_training
        self.reg = l2_reg
        self.in_h = in_h
        self.in_w = in_w

    def _encoder(self, x):
        x = tf.image.resize(x, size=(self.in_w, self.in_h),
                            method=tf.image.ResizeMethod.BICUBIC)
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        return x

    def _fusion(self, x, y):
        _, w, h, _ = x.shape
        y = tf.tile(y, (w, h, 1))
        y = tf.expand_dims(y, 0)
        x = Concatenate()([x, y])
        x = Conv2D(filters=256, kernel_size=1, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        return x

    def _decoder(self, x):
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2DTranspose(filters=128, kernel_size=3, strides=2,
                            padding='same',
                            activation='relu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=self.reg)(x)
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=1,
                            padding='same',
                            activation='relu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=self.reg)(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(x)
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                            padding='same',
                            activation='relu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=self.reg)(
            x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(
            x)
        x = Conv2D(filters=2, kernel_size=3, strides=1, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.reg)(
            x)
        x = Conv2DTranspose(filters=2, kernel_size=3, strides=2,
                            padding='same',
                            activation='tanh',
                            kernel_initializer='he_normal',
                            kernel_regularizer=self.reg)(
            x)
        return x

    def call(self, inputs, training=None, mask=None):
        x1 = self._encoder(inputs)
        x2 = InceptionResNetV2(include_top=False, pooling='avg')(inputs)
        x2 = tf.expand_dims(x2, 0)
        x = self._fusion(x1, x2)

        x = self._decoder(x)
        return x


if __name__ == "__main__":
    model = ColorizeGrayScale(is_training=True, l2_reg=l2(1e-3))

    image = tf.random.uniform(shape=(1, 229, 229, 3))
    output = model(image)
    print(output.shape)
