import os
import keras
from keras.models import model_from_json
import numpy as np
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
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

model.load_weights('weights/vanilla_conv_weights')

UPLOAD_FOLDER = 'cache'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_img = img.resize((450, 450))
            to_predict = np.asarray(new_img)

            img_predicted = model.predict(np.array([to_predict]))
            img_predicted = np.clip(img_predicted, 0, 255)

            to_save = Image.fromarray(img_predicted[0].astype(np.uint8))

            resized = 'resized_' + filename
            to_save.save(os.path.join(app.config['UPLOAD_FOLDER'], resized), "JPEG", optimize=True)
            return redirect(url_for('uploaded_file',
                                    filename=resized))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
