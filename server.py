from flask import Flask,request,jsonify

import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

client = vision.ImageAnnotatorClient()


app = Flask(__name__)
UPLOAD_FOLDER = '~/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST'])
def hello_world():
    request.files['photo'].save(request.files['photo'].filename)

    file_name = os.path.abspath('leaf.jpg')

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
        # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    print(labels)
    jsonthing={}
    for label in labels:
        jsonthing[label.description] = label.score
    return jsonify(jsonthing)
