import tensorflow as tf
from utils import rgb_to_lab, lab_to_rgb
from model import ColorizeGrayScale
from config import reg
from data import normalize_rgb
import matplotlib.pyplot as plt
from keras import backend as K

K.clear_session()
model = ColorizeGrayScale(l2_reg=reg)

model.load_weights('../model_weights/model.09')

# prepare the input
img_contents = tf.io.read_file('../images/2007_000033.jpg')
rgb = tf.image.decode_jpeg(img_contents)

# resize the image
input_1 = tf.image.resize_with_crop_or_pad(rgb, target_width=224,
                                           target_height=224)
input_2 = tf.image.resize_with_crop_or_pad(rgb, target_width=229,
                                           target_height=229)

# conver input_1 to lab
lab_img = rgb_to_lab(input_1)

l_channel = lab_img[:, :, 0]

# convert to grayscale
input_1 = tf.image.rgb_to_grayscale(input_1)
gray = input_1
input_2 = tf.image.rgb_to_grayscale(input_2)
input_2 = tf.stack([input_2[:, :, 0]] * 3, axis=2)

# Normalize the image [-1, 1]
input_1 = normalize_rgb(input_1)
input_2 = normalize_rgb(input_2)

# add batch dimension
input_1 = tf.expand_dims(input_1, axis=0)

inputs = {'input_1': input_1, 'input_2': input_2}

# run the model
output = model(inputs)

output = tf.squeeze(output, axis=0)

# Multiply output by 128
output = output * 128

# Add grayscale channel with output
lab_result = tf.stack(
    [tf.cast(l_channel, tf.float32), output[:, :, 0], output[:, :, 1]],
    axis=-1)

# convert lab result to rgb
rgb_colorized = lab_to_rgb(lab_result)

# plot the result
plt.imshow(rgb_colorized.numpy())
plt.show()
