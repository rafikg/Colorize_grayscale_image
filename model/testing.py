import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from utils import rgb_to_lab, lab_to_rgb
from model import ColorizeGrayScale
from config import reg
from data import normalize_rgb

from keras import backend as K

K.clear_session()

# load all images
image_files = glob.glob('../images/*.jpg')
batch_input_1 = []
batch_input_2 = []
batch_l_channel = []
# prepare the input
for file in image_files[4:]:
    img_contents = tf.io.read_file(file)
    rgb = tf.image.decode_jpeg(img_contents)
    # resize the image
    input_1 = tf.image.resize_with_crop_or_pad(rgb, target_width=224,
                                               target_height=224)
    input_2 = tf.image.resize_with_crop_or_pad(rgb, target_width=229,
                                               target_height=229)
    # conver input_1 to lab
    lab_img = rgb_to_lab(input_1)

    l_channel = lab_img[:, :, 0]
    batch_l_channel.append(l_channel)

    # convert to grayscale
    input_1 = tf.image.rgb_to_grayscale(input_1)
    input_2 = tf.image.rgb_to_grayscale(input_2)
    input_2 = tf.stack([input_2[:, :, 0]] * 3, axis=2)

    # Normalize the image [-1, 1]
    input_1 = normalize_rgb(input_1)
    input_2 = normalize_rgb(input_2)

    # add batch dimension
    input_1 = tf.expand_dims(input_1, axis=0)
    input_2 = tf.expand_dims(input_2, axis=0)
    batch_input_1.append(input_1)
    batch_input_2.append(input_2)
# reshape the input batch
batch_input_1 = tf.reshape(batch_input_1, shape=(-1, 224, 224, 1))
batch_input_2 = tf.reshape(batch_input_2, shape=(-1, 229, 229, 3))

inputs = {'input_1': batch_input_1, 'input_2': batch_input_2}

# build the model
model = ColorizeGrayScale(l2_reg=reg, is_training=False)

# upload the weights
model.load_weights('../model_weights/model.06')

# run the model
output = model(inputs)

# output = tf.squeeze(output, axis=0)

# Multiply output by 128
output = output * 128

# reshape batch_l_channel
batch_l_channel = tf.reshape(batch_l_channel, shape=(-1, 224, 224))
# Add l_channel channel with output
lab_result = tf.stack(
    [batch_l_channel[:, :, :], output[:, :, :, 0], output[:, :, :, 1]],
    axis=-1)

for idx in range(len(lab_result)):
    # convert lab result to rgb
    rgb_colorized = lab_to_rgb(lab_result[idx])
    # plot the result
    plt.imshow(rgb_colorized.numpy())
    plt.show()
