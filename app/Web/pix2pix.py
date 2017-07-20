import theano
import pickle
from lasagne.layers import get_output, set_all_param_values
from model.unet import build_UNet

class Generator:
    def __init__(self, weights):
        """

        :param weights: filename with model weights
        """

        with open(weights, "rb") as file:
            self.weights = pickle.load(file)

        self.model = build_UNet()
        self.model_input = self.model["input"].input_var
        self.model_output = get_output(self.model["output"])

        self.reconstruct = theano.function([self.model_input], self.model_output)

    def generate(self, sattelite_image):
        """
        :param sattelite_image: numpy array of shape [None, 3, 128, 128]
        :return: numpy array of shape [None, 3, 128, 128]
        """
        return self.reconstruct(sattelite_image)

    def gen_save(self, sattelite_image, fname):
        