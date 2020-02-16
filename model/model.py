from typing import Callable

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.regularizers import l2


class ColorizeGrayScale(Model):
    "Encapsulate the encoder-decoder network"

    def __init__(self, is_training: bool, l2_reg: Callable,
                 name: str = 'Colorize_grayscale'):

        super().__init__(name=name)
        self.is_training = is_training
        self.reg = l2_reg

        self.encod_conv1 = Conv2D(filters=64, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.encod_conv2 = Conv2D(filters=128, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.encod_conv3 = Conv2D(filters=128, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.encod_conv4 = Conv2D(filters=256, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.encod_conv5 = Conv2D(filters=256, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.encod_conv6 = Conv2D(filters=512, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)

        self.encod_conv7 = Conv2D(filters=512, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.encod_conv8 = Conv2D(filters=256, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)

        self.inception = InceptionResNetV2(include_top=False, pooling='avg')

        self.conv1x1 = Conv2D(filters=256, kernel_size=1, strides=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=self.reg)

        self.decod_conv1 = Conv2D(filters=128, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.decod_conv2 = Conv2DTranspose(filters=128, kernel_size=3,
                                           strides=2,
                                           padding='same',
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=self.reg)
        self.decod_conv3 = Conv2DTranspose(filters=64, kernel_size=3,
                                           strides=1,
                                           padding='same',
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=self.reg)
        self.decod_conv4 = Conv2D(filters=64, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.decod_conv5 = Conv2DTranspose(filters=64, kernel_size=3,
                                           strides=2,
                                           padding='same',
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=self.reg)
        self.decod_conv6 = Conv2D(filters=32, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.decod_conv7 = Conv2D(filters=2, kernel_size=3, strides=1,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=self.reg)
        self.decod_conv8 = Conv2DTranspose(filters=2, kernel_size=3, strides=2,
                                           padding='same',
                                           activation='tanh',
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=self.reg)

    def _encoder(self, x: tf.Tensor) -> tf.Tensor:
        """
        Encapsulate the encoder part of the model
        """
        x = self.encod_conv1(x)
        x = self.encod_conv2(x)
        x = self.encod_conv3(x)
        x = self.encod_conv4(x)
        x = self.encod_conv5(x)
        x = self.encod_conv6(x)
        x = self.encod_conv7(x)
        x = self.encod_conv8(x)
        return x

    def _fusion(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Encapsulate the fusion part of the model
        """
        _, w, h, _ = x.shape
        y = tf.tile(y, (1, w, h, 1))
        x = Concatenate()([x, y])
        x = self.conv1x1(x)
        return x

    def _decoder(self, x: tf.Tensor) -> tf.Tensor:
        """
        Encapsulate the decoder part of the model
        """
        x = self.decod_conv1(x)
        x = self.decod_conv2(x)
        x = self.decod_conv3(x)
        x = self.decod_conv4(x)
        x = self.decod_conv5(x)
        x = self.decod_conv6(x)
        x = self.decod_conv7(x)
        x = self.decod_conv8(x)
        return x

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        input1 = inputs['input_1']
        input2 = inputs['input_2']
        x1 = self._encoder(input1)
        x2 = self.inception(input2)
        x2 = tf.expand_dims(x2, axis=1)
        x2 = tf.expand_dims(x2, axis=1)
        x = self._fusion(x1, x2)

        x = self._decoder(x)
        return x


if __name__ == "__main__":
    model = ColorizeGrayScale(is_training=True, l2_reg=l2(1e-3))

    image = tf.random.uniform(shape=(1, 229, 229, 3))
    output = model(image)
    print(output.shape)
