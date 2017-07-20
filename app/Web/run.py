import requests
import json
import os
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
from model.neuro1 import get_prediction
from app.picparse import get_the_pictures

app = Flask(__name__)

@app.context_processor
def override_url_for():
    """
    Generate a new token on every request to prevent the browser from
    caching static files.
    """
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
    get_prediction()
    return render_template("index.html", imgname1='images/cat/sat.jpg', imgname2='images/cat/map.png', imgname3='images/cat/neuro1.jpg')

if __name__ == '__main__':
    app.run()