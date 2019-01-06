from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from render_26 import render_view

# Flask utils
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from extract_feature import extract_feature_from_list_imgs
from classify_model import predict

# Define a flask app
app = Flask(__name__)

print('Check http://35.199.180.29:5000/')


def id2label(id):
    lst = 'chair	light	pc	table	cup	storage	desk	bag	display	bookshelf	bin	book	oven	bed	box	pillow	machine	printer	sofa	keyboard'.split()
    return lst[id]

def model_predict(img_path):
    list_render_imgs = render_obj(img_path, real_render=False)
    print(list_render_imgs)
    X = extract_feature_from_list_imgs(list_render_imgs)
    scores,predict_label_id = predict(X)
    return scores, predict_label_id
    
def render_obj(obj_path,real_render):
    render_path = './static/'+obj_path+'/'
    if real_render:
        render_view(obj_path , render_path,0,26)
    list_render_imgs = [render_path+str(i)+'.png' for i in range(26)]
    
    return list_render_imgs

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def btn_upload():
    print('render called')
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        return file_path
    return None

@app.route('/render', methods=['GET', 'POST'])
def btn_render():
    print('render called')
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #render & return list img
        list_render_imgs = render_obj(file_path,real_render=True)
        return str(list_render_imgs)
    return None


@app.route('/predict', methods=['GET', 'POST'])
def btn_predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds_score, preds_id = model_predict(file_path)

        # Process your result for human
        pred_class_id = preds_score.argmax(axis=-1)            # Simple argmax
        
        #print('predicted:',pred_class_id)
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = id2label(int(pred_class_id[0]))              # Convert to string
        return result 
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
