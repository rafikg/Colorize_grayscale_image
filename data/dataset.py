from typing import List, Tuple, Callable

from pathlib import Path
from . import flip, crop_or_pad_image, get_gray_and_ab, normalize
import glob
import os
import tensorflow as tf


class ColorizeDataset(object):
    """
    Read images from the disk and enques them into a tensorflow queue using
    tf.data.Dataset
    """

    def __init__(self, path: str, img_ext: str,
                 in_height_br1: int = 224,
                 in_width_br1: int = 224,
                 in_height_br2: int = 229,
                 in_width_br2: int = 229,
                 batch_size: int = 1,
                 n_workers: int = 12,
                 is_cached: bool = False,
                 is_flip: bool = True,
                 is_training: bool = True,
                 is_validation: bool = False,
                 is_shuffle: bool = True,
                 name='Pascal2012'):

        self.path = Path(path)
        self.img_ext = img_ext
        self.in_height_br1 = in_height_br1
        self.in_width_br1 = in_width_br1
        self.in_height_br2 = in_height_br2
        self.in_width_br2 = in_width_br2
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.is_cached = is_cached
        self.is_flip = is_flip
        self.is_training = is_training
        self.is_validation = is_validation
        self.is_shuffle = is_shuffle

        self.name = name

        if not Path.exists(self.path):
            raise IOError(f"{self.path}' is invalid path")

        self.imgs_list = self.load_files_list()
        self.tf_data = self.create_dataset()

        # is_validation and is_training should be both true or both false
        xor = tf.math.logical_xor(
            self.is_validation,
            self.is_training,
            name='train_XOR_valid'
        )
        if not xor:
            raise ValueError(f'dataset should be created either'
                             f' for training or validation mode')

    def load_files_list(self) -> List:
        """
        Get all files paths inside the dataset path
        Returns
        -------

        """
        files_list = []
        for file in glob.glob(os.path.join(self.path, self.img_ext)):
            files_list.append(file)
        return files_list

    def image_decoding(self, x: dict) -> Tuple[dict, tf.Tensor]:
        """
        Decode the image and return the raw data
        Returns
        -------

        """
        file = x['input_1']
        img_contents = tf.io.read_file(file)

        # decode the image
        img = tf.image.decode_jpeg(img_contents, channels=3)
        img = tf.cast(img, dtype=tf.float64)
        x['input_1'] = img
        x['input_2'] = img
        y = img
        return x, y

    def create_dataset(self) -> Callable:
        """
        Create a tf.data.Dataset
        Returns
        -------
        tf.data.Dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            ({"input_1": self.imgs_list, "input_2": self.imgs_list},
             self.imgs_list))

        # load images
        dataset = dataset.map(lambda x, y: self.image_decoding(x),
                              num_parallel_calls=self.n_workers)
        # cached images
        if self.is_cached:
            dataset = dataset.cache()

        # shuffle images
        if self.is_shuffle:
            dataset = dataset.shuffle(buffer_size=128)

        # flip images horizontally
        if self.is_flip and not self.is_validation:
            if tf.random.uniform(shape=(1,)) > 0.5:
                dataset = dataset.map(lambda x, y: flip(x),
                                      num_parallel_calls=self.n_workers)

        # convert rgb to lab and return
        dataset = dataset.map(lambda x, y: get_gray_and_ab(x),
                              num_parallel_calls=self.n_workers)

        # crop or pad images
        dataset = dataset.map(
            lambda x, y: crop_or_pad_image(x=x, y=y,
                                           in_h_br1=self.in_height_br1,
                                           in_w_br1=self.in_width_br1,
                                           in_h_br2=self.in_height_br2,
                                           in_w_br2=self.in_width_br2),
            num_parallel_calls=self.n_workers)

        # Normalize data between [-1, 1]
        dataset = dataset.map(lambda x, y: normalize(x, y))

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=1)

        return dataset
