import theano
import pickle
import matplotlib.pyplot as plt
from lasagne.layers import get_output, set_all_param_values
from model.training.unet import build_UNet
from model.training.preproc import pics2array

class Generator:
    def __init__(self, weights="model/weights/pix2pix_weights.pcl"):
        """
        :param weights: filename with models weights
        """

        with open(weights, "rb") as file:
            self.weights = pickle.load(file)

        self.model = build_UNet()
        self.model_input = self.model["input"].input_var
        self.model_output = get_output(self.model["output"])

        self.reconstruct = theano.function([self.model_input], self.model_output)

    def generate(self, sattelite_image):
        """
        :param sattelite_image: numpy array of shape [None, 3, 200, 200]
        :return: numpy array of shape [None, 3, 200, 200]
        """
        return self.reconstruct(sattelite_image)

    def gen_save(self, sattelite_image, fname):
        """
        :param sattelite_image: path to sattelite image
        :param fname: file ti save transforemed pic
        """

        sattelite_image = pics2array("", sattelite_image, [200, 200])
        reconstructed = self.reconstruct(sattelite_image[0 : 1])

        plt.imsave(fname, reconstructed)

gen = Generator()