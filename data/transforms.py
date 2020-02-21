from typing import Tuple

import tensorflow as tf
from utils import rgb_to_lab


def rgb_to_gray(image: tf.Tensor) -> tf.Tensor:
    """
    Convert RGB image to gray scale image
    Parameters
    ----------
    image : tf.tensor

    Returns
    -------

    """
    gray = tf.image.rgb_to_grayscale(image)
    return gray


def gray_to_rgb(image: tf.Tensor) -> tf.Tensor:
    """
    Just stack the grayscale 3 times
    Parameters
    ----------
    image : tf.tensor

    Returns
    -------
    rgb image: tf.tensor
    """
    rgb = tf.stack([image] * 3, axis=-1)
    return rgb


def normalize_rgb(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize rgb or grayscale image between [-1, 1]
    Parameters
    ----------
    image : tf.tensor

    Returns
    -------
    normalized image : tf.tensor
    """
    min_x = tf.reduce_min(image)
    max_x = tf.reduce_max(image)

    image = (image - min_x) / (max_x - min_x)
    return 2 * image - 1


def normalize_ab(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize  ab channels between [-1, 1]
    ab is between [-127, 128]
    Parameters
    ----------
    image : tf.tensor

    Returns
    -------
    normalized image: tf.tensor
    """
    image = image / 128
    return image


def normalize_l(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize l channel between [-1, 1]
    l is between 0 and 100
    Parameters:
    ----------
    image: tf.Tensor

    Returns:
    -------
    normalized image: tf.Tensor
    """
    image = 2 * image / 100 - 1
    return image


def get_l_channel(image: tf.Tensor) -> tf.Tensor:
    """
    Get the channel l of the LAB image
    Parameters
    ----------
    image: tf.Tensor
        image in the LAB color space

    Returns
    -------
    lightness (channel l): tf.Tensor
    """

    l, _, _ = tf.unstack(image, axis=-1)
    return tf.expand_dims(l, axis=-1)


def get_gray_and_ab(image: dict) -> Tuple[dict, tf.Tensor]:
    """
    Get grayscale and ab channels from an rgb image
    Parameters
    ----------
    image: tf.Tensor
        image in the LAB color space

    Returns
    -------
    ab channels: tf.Tensor
    """
    img = image['input_1']
    gray = rgb_to_gray(img)
    lab = rgb_to_lab(img)
    ab = lab[:, :, 1:]
    image['input_1'] = gray
    image['input_2'] = gray
    return image, ab
