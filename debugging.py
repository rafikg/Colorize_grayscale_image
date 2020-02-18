import matplotlib.pyplot as plt
from data import ColorizeDataset
from utils import rgb_to_lab, lab_to_rgb
import skimage.io as io
import skimage.color as color

def plot_img(x, y):
    f, axarr = plt.subplots(2, 2)
    img1 = x['input_1']
    img2 = x['input_2']

    axarr[0, 0].imshow(img1[0, :, :, 0])
    axarr[0, 1].imshow(img2[0])
    axarr[1, 0].imshow(y[0, :, :, 0])
    axarr[1, 1].imshow(y[0, :, :, 1])
    plt.show()


# dataObject = ColorizeDataset(path='dataset/train_data', img_ext="*.jpg")
# data = dataObject.tf_data
# for x, y in data:
#     print(x['input_1'].shape, x['input_2'].shape, y.shape)
#     # plot_img(x, y)


image = io.imread("./images/index.jpeg")

ski_lab = color.rgb2lab(image)

print("ski_lab min_max", ski_lab.min(), ski_lab.max())

lab = rgb_to_lab(image).numpy()
print("lab min_max ", lab.min(), lab.max())

rgb = lab_to_rgb(lab)

plt.imshow(rgb)
plt.show()