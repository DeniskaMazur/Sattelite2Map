import os
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
# from model.neuro1 import get_prediction
from picparse import get_the_pictures
# from convolutional_net import ConvNet
app = Flask(__name__)
model = ConvNet('net3.h5')
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/')
def my_form():
    return render_template("index.html", imgname1='images/map1.png', imgname2='images/map1.png', imgname3='images/map1.png')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    get_the_pictures(text)
    model.gen_save('static/images/cat/sat.jpg', 'static/images/cat/neuro1.jpg')
    return render_template("index.html", imgname1='images/cat/sat.jpg', imgname2='images/cat/map.png', imgname3='images/cat/neuro1.jpg')

if __name__ == '__main__':
    app.run()