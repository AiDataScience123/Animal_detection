import io
import os
import json
import wikipediaapi
import requests
import PIL
from PIL import Image
import argparse
import torch
from bs4 import BeautifulSoup
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")

model_name = find_model()
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    results = model(imgs, size=640)
    return results

import requests
import wikipediaapi

def get_animal_info(animal_name, user_agent):
    url = 'https://en.wikipedia.org/wiki/' + animal_name
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        
        # Concatenate the text from the first few paragraphs
        text = ''.join([p.text for p in paragraphs[:3]])
        return text[:1000]
    else:
        return 'Failed to retrieve information from Wikipedia'



@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return redirect(request.url)
            
            file = request.files.get('file')
            if not file:
                return render_template('error.html', error_message="No file provided")

            img_bytes = file.read()
            results = get_prediction(img_bytes)

            if results is not None:
                animal_name = results.names[int(results.xyxy[0][0][-1])] if results.xyxy[0].shape[0] > 0 else "Not detected"
                animal_info = get_animal_info(animal_name,'AnimalSpeciesInformation')
                print(animal_info)
                results.save(save_dir='static')
                filename = 'image0.jpg'
                return render_template('result.html', result_image=filename, model_name=model_name,
                                       animal_name=animal_name, animal_info=animal_info)
            else:
                return render_template('error.html', error_message="Error during prediction")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return render_template('error.html', error_message="An unexpected error occurred")

    return render_template('index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
