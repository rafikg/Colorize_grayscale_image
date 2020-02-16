from typing import Tuple

import tensorflow as tf

from . import normalize_rgb, normalize_ab, gray_to_rgb


def crop_or_pad_image(x: dict, y: tf.Tensor,
                      in_h_br1: int, in_w_br1: int,
                      in_h_br2: int, in_w_br2: int) -> Tuple[
                                                       dict, tf.Tensor]:
    """
        Random crop or pad x and y image with zero
        Parameters
        ----------
        x: dict
        y: tf.Tensor
        in_h_br1: int
        in_w_br1: int
        in_h_br2:int
        in_w_br2:int
        Returns
        -------
        x, y: Tuple(dict, tf.Tensor)
        """
    img = x['input_1']

    x_shape = img.shape
    last_dim_x = x_shape[-1]

    concat = tf.concat([img, y], axis=-1)

    concat_crop = tf.image.resize_with_crop_or_pad(
        image=concat,
        target_height=in_h_br1,
        target_width=in_w_br1
    )
    # input_1
    x1_crop = concat_crop[:, :, :last_dim_x]

    # output
    y_crop = concat_crop[:, :, last_dim_x:]

    # input_2
    x2_crop = tf.cast(tf.image.resize(img, size=(in_h_br2,
                                                 in_w_br2),
                                      method=tf.image.ResizeMethod.BICUBIC),
                      tf.float64)

    # convert input_2 to 3-channels grayscale
    x2_crop = tf.squeeze(x2_crop, axis=-1)
    x2_crop = gray_to_rgb(x2_crop)

    x['input_1'] = x1_crop
    x['input_2'] = x2_crop
    y = y_crop
    return x, y


def flip(x: dict) -> Tuple[dict, tf.Tensor]:
    """
    Flip horizontally x image
    Parameters
    ----------
    x : dict

    Returns
    -------
    Flipped images Tuple(tf.Tensor, tf.Tensor)
    """
    img = x['input_1']
    img = tf.image.flip_left_right(img)
    x['input_1'] = img
    x['input_2'] = img
    y = img
    return x, y


def normalize(x: dict, y: tf.Tensor) -> Tuple[
                                        dict, tf.Tensor]:
    """
    Normalize rgb and ab image between [-1, 1]
    Parameters
    ----------
    x : dict
    y: tf.tensor

    Returns
    -------
    normalized images Tuple(dict, tf.Tensor)
    """
    img_1 = x['input_1']
    img_2 = x['input_2']
    img_1 = normalize_rgb(img_1)
    img_2 = normalize_rgb(img_2)
    y = normalize_ab(y)
    x['input_1'] = img_1
    x['input_2'] = img_2
    return x, y
