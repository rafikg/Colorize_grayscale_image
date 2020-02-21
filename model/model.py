from typing import Callable

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, \
    UpSampling2D
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.regularizers import l2

from data import ColorizeDataset


class ColorizeGrayScale(Model):
    "Encapsulate the encoder-decoder network"

    def __init__(self, l2_reg: Callable,
                 name: str = 'Colorize_grayscale'):
        super().__init__(name=name)
        self.reg = l2_reg
        self.encod_conv1 = Conv2D(filters=64, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv1')

        self.encod_conv2 = Conv2D(filters=128, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv2')

        self.encod_conv3 = Conv2D(filters=128, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv3')

        self.encod_conv4 = Conv2D(filters=256, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv4')

        self.encod_conv5 = Conv2D(filters=256, kernel_size=3, strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv5')

        self.encod_conv6 = Conv2D(filters=512, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv6')

        self.encod_conv7 = Conv2D(filters=512, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv7')

        self.encod_conv8 = Conv2D(filters=256, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='encod_conv8')

        self.inception = InceptionResNetV2(include_top=False, pooling='avg')
        # freeze all layers for inception:
        for layer in self.inception.layers:
            layer.trainable = False
        self.conv1x1 = Conv2D(filters=256, kernel_size=1, strides=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=self.reg)

        self.decod_conv1 = Conv2D(filters=128, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv1')
        self.decod_conv2 = UpSampling2D()
        self.decod_conv3 = Conv2D(filters=64, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv3')
        self.decod_conv4 = Conv2D(filters=64, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv4')
        self.decod_conv5 = UpSampling2D()
        self.decod_conv6 = Conv2D(filters=32, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv6')
        self.decod_conv7 = Conv2D(filters=2, kernel_size=3,
                                  padding='same',
                                  activation='tanh',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=self.reg,
                                  name='decod_conv7')
        self.decod_conv8 = UpSampling2D()

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
    BATCH_SIZE = 1
    model = ColorizeGrayScale(l2_reg=l2(1e-3))

    # dataset_train_obj = ColorizeDataset(
    #     path="../dataset/train_data",
    #     img_ext="*.jpg",
    #     batch_size=BATCH_SIZE,
    #     n_workers=12)
    # from utils import rgb_to_lab, lab_to_rgb
    # import skimage.color as color
    #
    # data_set = dataset_train_obj.tf_data
    # for x, y in data_set:
    #     ski_lab = color.rgb2lab(x['input_1'])
    #     lab = rgb_to_lab(x['input_1'])
    #     print("ski_lab min_max", ski_lab.min(), ski_lab.max())
    #     print("lab min_max", lab.numpy().min(), lab.numpy().max())
