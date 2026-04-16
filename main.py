import numpy as np
from mnist import MNIST

# Loading data from files
mndata = MNIST('data')
images, labels = mndata.load_training()

images = np.array(images)
labels = np.array(labels)

m, n = images.shape
print(m, n)


#print(mndata.display(images[0]))