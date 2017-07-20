from keras.models import model_from_json
import numpy as np
from PIL import Image

model = model_from_json('{"class_name": "Sequential", "config": [{"class_name": "Conv2D", "config": {"name": '
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

model.load_weights('model/map_net_weights_526')


def get_prediction(path='static/images/cat/sat.jpg'):
    img = Image.open(path)
    new_img = img.resize((450, 450))
    to_predict = np.asarray(new_img)
    img_predicted = model.predict(np.array([to_predict]))
    img_predicted = np.clip(img_predicted, 0, 255)
    to_save = Image.fromarray(img_predicted[0].astype(np.uint8))
    to_save_res = to_save.resize((200, 200))
    to_save_res.save('static/images/cat/neuro1.jpg', "JPEG", optimize=True)