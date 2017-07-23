import numpy as np
import keras
from scipy import ndimage
import matplotlib.pyplot as plt
# import os # Uncomment if GPU is available
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ConvNet:
    def __init__(self, weights):
        self.model = keras.models.load_model(weights)

    def gen_save(self, satellite_image, fname):
        img = ndimage.imread(satellite_image)
        img = ndimage.zoom(img, (400 / img.shape[0], 400 / img.shape[1], 1))
        img = img / 255
        predicted = self.model.predict(np.array([img]))
        plt.imsave(fname, predicted[0])

