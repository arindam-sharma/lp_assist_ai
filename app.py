from flask import Flask, request, render_template, redirect, url_for, send_from_directory, after_this_request
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from model.model import CustomResNet
from werkzeug.utils import secure_filename
import os 

import __main__
setattr(__main__, "CustomResNet", CustomResNet)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


model = torch.load('model/saved_model.pt',map_location=torch.device('cpu'))
model.eval()
#image = Image.open(img_path).convert('RGB')

class_names = ['Bad Quality', 'Spinal Cord Present', 'Fluid Present']

def pre_process_image(image_path):

    image = Image.open(image_path).convert('RGB')
    transformation = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])
    image_tensor = transformation(image).unsqueeze(0)
    return image_tensor


@app.route('/predict', methods = ['POST'])
def predict():
    image = request.files['image']

    if image.filename == '':
        return redirect(request.url)  # Redirect if no file is selected

    # Assuming the file is valid, save it
    filename = secure_filename(image.filename)  
    filepath = os.path.join('static/uploads', filename)
    image.save(filepath)
    
 
    image_tensor = pre_process_image(filepath)
    output = model(image_tensor)
    print(output)
    predicts = (output > 0.5).float()
    print(predicts)
    predicts = predicts.flatten()
    class_index = [i for i in range(0, 3) if predicts[i] == 1]
    predicted_class = [class_names[j] for j in class_index]
    predicted_classes = ' and '.join(predicted_class)


    return render_template('predict.html', predicted_class=predicted_classes,
                           image_filename=filename)


# add disclaimers on the homepaage
# work on securing the photo/image


