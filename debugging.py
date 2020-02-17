import matplotlib.pyplot as plt
from data import ColorizeDataset


def plot_img(x, y):
    f, axarr = plt.subplots(2, 2)
    img1 = x['input_1']
    img2 = x['input_2']

    axarr[0, 0].imshow(img1[0, :, :, 0])
    axarr[0, 1].imshow(img2[0])
    axarr[1, 0].imshow(y[0, :, :, 0])
    axarr[1, 1].imshow(y[0, :, :, 1])
    plt.show()


dataObject = ColorizeDataset(path='dataset/train_data', img_ext="*.jpg")
data = dataObject.tf_data
for x, y in data:
    print(x['input_1'].shape, x['input_2'].shape, y.shape)
    # plot_img(x, y)
