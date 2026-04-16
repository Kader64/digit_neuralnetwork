import numpy as np
from mnist import MNIST

# Loading data from files
mndata = MNIST('data')
images, labels = mndata.load_training()

images = np.array(images)
labels = np.array(labels)
countImg, sizeImg = images.shape

# Splitting data into validation and training set
data_val = images[0:1000]
label_val = labels[0:1000]
data_val = data_val / 255

data_train = images[1000:]
label_train = labels[1000:]
data_train = data_train / 255


