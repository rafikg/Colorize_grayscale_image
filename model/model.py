from typing import Callable

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, \
    UpSampling2D, BatchNormalization
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


class ColorizeGrayScale(Model):
    "Encapsulate the encoder-decoder network"

    def __init__(self, l2_reg: Callable,
                 name: str = 'Colorize_grayscale', is_training:bool=True):
        super().__init__(name=name)
        self.reg = l2_reg
        self.is_training = is_training
        self.encod_conv1 = Conv2D(filters=64, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv1')
        self.en1_bn = BatchNormalization()

        self.encod_conv2 = Conv2D(filters=128, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv2')
        self.en2_bn = BatchNormalization()

        self.encod_conv3 = Conv2D(filters=128, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv3')
        self.en3_bn = BatchNormalization()

        self.encod_conv4 = Conv2D(filters=256, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv4')
        self.en4_bn = BatchNormalization()

        self.encod_conv5 = Conv2D(filters=256, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv5')
        self.en5_bn = BatchNormalization()

        self.encod_conv6 = Conv2D(filters=512, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv6')
        self.en6_bn = BatchNormalization()

        self.encod_conv7 = Conv2D(filters=512, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv7')
        self.en7_bn = BatchNormalization()

        self.encod_conv8 = Conv2D(filters=256, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv8')
        self.en8_bn = BatchNormalization()

        self.inception = InceptionResNetV2(include_top=False, pooling='avg')
        # freeze all layers for inception:
        for layer in self.inception.layers:
            layer.trainable = False
        self.conv1x1 = Conv2D(filters=256, kernel_size=1, strides=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=self.reg)
        self.bn = BatchNormalization()

        self.decod_conv1 = Conv2D(filters=128, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv1')
        self.dec1_bn = BatchNormalization()
        self.upsamp_1 = UpSampling2D()
        self.decod_conv3 = Conv2D(filters=64, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv3')
        self.dec2_bn = BatchNormalization()
        self.decod_conv4 = Conv2D(filters=64, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv4')
        self.dec3_bn = BatchNormalization()
        self.upsamp_2 = UpSampling2D()
        self.decod_conv6 = Conv2D(filters=32, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv6')
        self.dec4_bn = BatchNormalization()
        self.decod_conv7 = Conv2D(filters=2, kernel_size=3,
                                  padding='same',
                                  activation='tanh',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv7')
        self.dec5_bn = BatchNormalization()
        self.upsamp_3 = UpSampling2D()

    def _encoder(self, x: tf.Tensor) -> tf.Tensor:
        """
        Encapsulate the encoder part of the model
        """
        x = self.encod_conv1(x)
        x = self.en1_bn(x, training=self.is_training)
        x = self.encod_conv2(x)
        x = self.en2_bn(x, training=self.is_training)
        x = self.encod_conv3(x)
        x = self.en3_bn(x, training=self.is_training)
        x = self.encod_conv4(x)
        x = self.en4_bn(x, training=self.is_training)
        x = self.encod_conv5(x)
        x = self.en5_bn(x, training=self.is_training)
        x = self.encod_conv6(x)
        x = self.en6_bn(x, training=self.is_training)
        x = self.encod_conv7(x)
        x = self.en7_bn(x, training=self.is_training)
        x = self.encod_conv8(x)
        x = self.en8_bn(x, training=self.is_training)
        return x

    def _fusion(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Encapsulate the fusion part of the model
        """
        _, w, h, _ = x.shape
        y = tf.tile(y, (1, w, h, 1))
        x = Concatenate()([x, y])
        x = self.conv1x1(x)
        x = self.bn(x, training=self.is_training)
        return x

    def _decoder(self, x: tf.Tensor) -> tf.Tensor:
        """
        Encapsulate the decoder part of the model
        """
        x = self.decod_conv1(x)
        x = self.dec1_bn(x, training=self.is_training)
        x = self.upsamp_1(x)
        x = self.decod_conv3(x)
        x = self.dec2_bn(x, training=self.is_training)
        x = self.decod_conv4(x)
        x = self.dec3_bn(x, training=self.is_training)
        x = self.upsamp_2(x)
        x = self.decod_conv6(x)
        x = self.dec4_bn(x, training=self.is_training)
        x = self.decod_conv7(x)
        x = self.dec5_bn(x, training=self.is_training)
        x = self.upsamp_3(x)
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
