import numpy as np
import keras
from PIL import Image
# import os # Uncomment if GPU is available
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ConvNet:
    def __init__(self, weights):
        self.model = keras.models.model_from_json(
                        '{"class_name": "Sequential", "config": [{"class_name": "Conv2D", "config": {"name": '
                        '"conv2d_1", "trainable": true, "batch_input_shape": [null, 450, 450, 3], "dtype": "float32", '
                        '"filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": '
                        '"channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, '
                        '"kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, '
                        '"mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {'
                        '"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, '
                        '"activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, '
                        '{"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, '
                        '"batch_input_shape": [null, 450, 450, 3], "dtype": "float32", "filters": 64, "kernel_size": '
                        '[3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", '
                        '"dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {'
                        '"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": '
                        '"uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", '
                        '"config": {"name": "batch_normalization_1", "trainable": true, "axis": -1, "momentum": 0.99, '
                        '"epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", '
                        '"config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, '
                        '"moving_mean_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": '
                        'null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, '
                        '{"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "filters": 64, '
                        '"kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", '
                        '"dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {'
                        '"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": '
                        '"uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", '
                        '"config": {"name": "batch_normalization_2", "trainable": true, "axis": -1, "momentum": 0.99, '
                        '"epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", '
                        '"config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, '
                        '"moving_mean_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": '
                        'null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, '
                        '{"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "filters": 64, '
                        '"kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", '
                        '"dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {'
                        '"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": '
                        '"uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {'
                        '"name": "dropout_1", "trainable": true, "rate": 0.25}}, {"class_name": "Conv2D", "config": {'
                        '"name": "conv2d_5", "trainable": true, "filters": 64, "kernel_size": [3, 3], "strides": [1, '
                        '1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], '
                        '"activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": '
                        '"VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", '
                        '"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", '
                        '"config": {"name": "batch_normalization_3", "trainable": true, "axis": -1, "momentum": 0.99, '
                        '"epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", '
                        '"config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, '
                        '"moving_mean_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": '
                        'null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, '
                        '{"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "filters": 64, '
                        '"kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", '
                        '"dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {'
                        '"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": '
                        '"uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", '
                        '"config": {"name": "batch_normalization_4", "trainable": true, "axis": -1, "momentum": 0.99, '
                        '"epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", '
                        '"config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, '
                        '"moving_mean_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": '
                        'null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, '
                        '{"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "filters": 64, '
                        '"kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", '
                        '"dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {'
                        '"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": '
                        '"uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {'
                        '"name": "conv2d_8", "trainable": true, "filters": 3, "kernel_size": [3, 3], "strides": [1, '
                        '1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], '
                        '"activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": '
                        '"VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", '
                        '"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, '
                        '"kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, '
                        '"kernel_constraint": null, "bias_constraint": null}}], "keras_version": "2.0.6", "backend": '
                        '"tensorflow"}')
        self.model.load_weights(weights)

    def gen_save(self, satellite_image, fname):
        img = Image.open(satellite_image)
        img = img.resize((450, 450))
        img = np.asarray(img)
        predicted = self.model.predict(np.array([img]))
        predicted = np.clip(predicted, 0, 255)
        img = Image.fromarray(predicted[0].astype(np.uint8))
        img.save(fname, "JPEG", optimize=True)
