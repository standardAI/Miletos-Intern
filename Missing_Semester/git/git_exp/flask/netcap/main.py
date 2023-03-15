from flask import Flask, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from captum.attr import GuidedBackprop, Occlusion, Saliency
import json



# Class index to class name mapping
labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


# Replaces all layers of a given type in the model with a new layer
# This is required to replace ReLU layers with ReLU(inplace=False) layers
# for Guided Backpropagation calculations
def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)  # Replace model's n attribute with new

# Define the transforms for the data; why these numbers?
data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

# For reproducible results
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #np.random.seed(seed)
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(0)


UPLOAD_FOLDER = '/home/cosmos/Documents/Miletos/Missing_Semester/git/git_exp/flask/netcap/static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)  # This is needed so that Flask knows where to look for resources such as templates and static files.


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def selectibles_w_upload():
  selectibles = '''
                          <!doctype html>
                          <title>Upload new File</title>
                          <h1>Upload new File</h1>
                          <style>
                          * {
                            box-sizing: border-box;
                          }
                          .column {
                            width: 100%;
                            padding: 10px;
                            height: 50px;
                          }
                          </style>
                          <section>
                          <form method=post enctype=multipart/form-data>
                            <input type=file name=file>
                            <input type=submit value=Upload>
                          </form>
                          <form action="/results" method="get">
                            <div class="column" style="background-color:#aaa;">
                                <span>Please select Neural Network type</span>
                                <input type="checkbox" id="net1" name="net1" value="vgg">
                                <label for="net1"> VGGNet</label>
                                <input type="checkbox" id="net2" name="net2" value="resnet">
                                <label for="net2"> ResNet50</label>
                                <span> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Please select Visualization type</span>
                                <input type="checkbox" id="viz1" name="viz1" value="guidedbackprop">
                                <label for="viz1"> Guided Backprop</label>
                                <input type="checkbox" id="viz2" name="viz2" value="saliency">
                                <label for="viz2"> Saliency</label>
                                <input type="checkbox" id="viz3" name="viz3" value="occlusion">
                                <label for="viz3"> Occlusion&emsp;&emsp;&emsp;&emsp;&emsp;</label>
                                <input type="submit" value="Submit">
                            </div>
                          '''
  return selectibles

def first_page():
  uploaded_files = selectibles_w_upload()
  uploaded_files += '<h1>Uploaded Files</h1>'
  for i, file in enumerate(os.listdir(UPLOAD_FOLDER + '/uploads')):
      uploaded_files += '<input type="checkbox" id="myCheckbox' + str(i+1) + '" name="myCheckbox' + str(i+1) + '" value="' + file + '">'
      uploaded_files += '<label for="myCheckbox' + str(i+1) + '"><img src="/static/uploads/' + file + '" width="200" height="200" /></label>'
  uploaded_files += '</form></section>'

  return uploaded_files

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER + "/uploads", filename))
            
            uploaded_files = first_page()

            return uploaded_files

    uploaded_files = first_page()

#    if os.listdir(UPLOAD_FOLDER + '/results'):
#      for file in os.listdir(UPLOAD_FOLDER + '/results'):
#        os.remove(UPLOAD_FOLDER + '/results/' + file)    

    return uploaded_files


@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'GET':
        net1 = request.args.get('net1')
        net2 = request.args.get('net2')
        viz1 = request.args.get('viz1')
        viz2 = request.args.get('viz2')
        viz3 = request.args.get('viz3')
        myCheckboxes = []
        for i in range(1, 101):  # 100 is the max number of selected images
            myCheckboxes.append(request.args.get('myCheckbox' + str(i)))
        #print(net1, net2, viz1, viz2, viz3, myCheckboxes)

        

        get_html = '''
                          <!doctype html>
                          <title>Results</title>
                          <h1>Results</h1>
                              <a href="/" class="button">Go to Main Page</a>
                          <style>
                          * {
                            box-sizing: border-box;
                          }
                          .column {
                            width: 100%;
                            padding: 10px;
                            height: 50px;
                          }
                          </style>
                          <section>
                            <div class="column" style="background-color:#aaa;">
                                <form action="/results" method=["GET", "POST"]>
                                <span>Please select Neural Network type</span>
                                <input type="checkbox" id="net1" name="net1" value="vgg">
                                <label for="net1"> VGGNet</label>
                                <input type="checkbox" id="net2" name="net2" value="resnet">
                                <label for="net2"> ResNet50</label>
                                <span> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Please select Visualization type</span>
                                <input type="checkbox" id="viz1" name="viz1" value="guidedbackprop">
                                <label for="viz1"> Guided Backprop</label>
                                <input type="checkbox" id="viz2" name="viz2" value="saliency">
                                <label for="viz2"> Saliency</label>
                                <input type="checkbox" id="viz3" name="viz3" value="occlusion">
                                <label for="viz3"> Occlusion&emsp;&emsp;&emsp;&emsp;&emsp;</label>
                                <input type="submit" value="Submit">
                                
                          '''

        i = 0
        for file in myCheckboxes:
          if file:
            get_html += '<input type="hidden" id="myCheckbox' + str(i+1) + '" name="myCheckbox' + str(i+1) + '" value="' + file + '">'
            get_html += '<label for="myCheckbox' + str(i+1) + '"></label>'
            i += 1

        get_html += '</form></div></section>'

        key = ''
        # Key Creation
        if net1:
          key += 'vgg'
        if net2:
          key += 'resnet'
        if viz1:
          key += 'guidedbackprop'
        if viz2:
          key += 'saliency'
        if viz3:
          key += 'occlusion'

        # If the key is not in the keys file:
        if net1 == 'vgg':
            if viz1 == 'guidedbackprop':
                viz_results = visualization(net1, viz1, myCheckboxes)
                
            if viz2 == 'saliency':
                viz_results = visualization(net1, viz2, myCheckboxes)

            if viz3 == 'occlusion':
                viz_results = visualization(net1, viz3, myCheckboxes)

        if net2 == 'resnet':
            if viz1 == 'guidedbackprop':
                viz_results = visualization(net2, viz1, myCheckboxes)

            if viz2 == 'saliency':
                viz_results = visualization(net2, viz2, myCheckboxes)

            if viz3 == 'occlusion':
                viz_results = visualization(net2, viz3, myCheckboxes)


        if os.listdir(UPLOAD_FOLDER + '/results'):
          if net1 == 'vgg':
            get_html += '<h1>Results for VGGNet</h1>'
            get_html += '<table>'
            if myCheckboxes[0] != None:
                get_html += '<tr>'
                get_html += '<td><h3 style="text-align:center;">Original Image</h3></td>'
                if viz1 == 'guidedbackprop':
                  get_html += '<td><h3 style="text-align: center;">Guided Backprop</h3></td>'
                if viz2 == 'saliency':
                  get_html += '<td><h3 style="text-align: center;">Saliency</h3></td>'
                if viz3 == 'occlusion':
                  get_html += '<td><h3 style="text-align: center;">Occlusion</h3></td>'
                get_html += '</tr>'

            for file in myCheckboxes:
              if file:
                get_html += '<tr>'
                get_html += '<td><img src="/static/uploads/' + file + '" width="200" height="200" /></td>'
                if viz1 == 'guidedbackprop':
                  get_html += '<td><img src="/static/results/' + file[:-4] + '_guidedbackprop_vgg.jpg" width="200" height="200" /></td>'
                  if viz2 is None:
                    get_html += '</tr>'
                if viz2 == 'saliency':
                  get_html += '<td><img src="/static/results/' + file[:-4] + '_saliency_vgg.jpg" width="200" height="200" /></td>'
                  if viz3 is None:
                    get_html += '</tr>'
                if viz3 == 'occlusion':
                  get_html += '<td><img src="/static/results/' + file[:-4] + '_occlusion_vgg.jpg" width="200" height="200" /></td>'
                  get_html += '</tr>'
            get_html += '</table>'
          
          if net2 == 'resnet':
            get_html += '<h1>Results for ResNet50</h1>'
            get_html += '<table>'
            if myCheckboxes[0] != None:
                get_html += '<tr>'
                get_html += '<td><h3 style="text-align:center;">Original Image</h3></td>'
                if viz1 == 'guidedbackprop':
                  get_html += '<td><h3 style="text-align: center;">Guided Backprop</h3></td>'
                if viz2 == 'saliency':
                  get_html += '<td><h3 style="text-align: center;">Saliency</h3></td>'
                if viz3 == 'occlusion':
                  get_html += '<td><h3 style="text-align: center;">Occlusion</h3></td>'
                get_html += '</tr>'

            for file in myCheckboxes:
              if file:
                get_html += '<tr>'
                get_html += '<td><img src="static/uploads/' + file + '" width="200" height="200" /></td>'
                if viz1 == 'guidedbackprop':
                  get_html += '<td><img src="static/results/' + file[:-4] + '_guidedbackprop_resnet.jpg" width="200" height="200" /></td>'
                  if viz2 is None:
                    get_html += '</tr>'
                if viz2 == 'saliency':
                  get_html += '<td><img src="static/results/' + file[:-4] + '_saliency_resnet.jpg" width="200" height="200" /></td>'
                  if viz3 is None:
                    get_html += '</tr>'
                if viz3 == 'occlusion':
                  get_html += '<td><img src="static/results/' + file[:-4] + '_occlusion_resnet.jpg" width="200" height="200" /></td>'
                  get_html += '</tr>'
            get_html += '</table>'
    
        return get_html


def visualization_inner(vi, model, checkBoxes, occlusion):
  attrs = []
  files = []

  for file in checkBoxes:
    if file is not None:
        image = Image.open(UPLOAD_FOLDER + '/uploads/' + file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        output = model(image)
        probs, idx = output.data.cpu().topk(1)
        output_softmax = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output_softmax, 5)
        pred_label_idx.squeeze_()
        predicted_label = [idx_to_labels[str(i.item())][1] for i in pred_label_idx]
        if occlusion:  # Occlusion needs sliding_window_shapes and strides parameters
          # TODO: Check this, cuz this might be wrong
          attr = vi.attribute(image, target=idx[0][0], sliding_window_shapes=(3, 15, 15), strides=(3, 8, 8), baselines=0)
        else:
          attr = vi.attribute(image, target=idx[0][0])
        attr = attr.squeeze().cpu().detach().numpy()
        attr = np.transpose(attr, (1, 2, 0))  # (C, H, W) -> (H, W, C) for PIL image
        attr = np.abs(attr)  # Take absolute value of gradients to enhance the visualization
        attr = np.max(attr, axis=2)  # Take the maximum over the channels
        attr = np.clip(attr, 0, 1)  # Clip to 0-1 range ->  [-1,0,1,2] -> [0,0,1,1]
        attr = np.uint8(attr * 255)  # Scale to 0-255 range
        attr = Image.fromarray(attr)
        attr = attr.resize((224, 224))
        attrs.append(attr)
        files.append(file)
  
  return attrs, files


def visualization(net, viz, checkBoxes):
    if net == 'vgg':
      model = models.vgg16(pretrained=True)
      model.eval()
      model = model.cuda()
      if viz == 'guidedbackprop':
          replace_layers(model, nn.ReLU, nn.ReLU(inplace=False))
          vi = GuidedBackprop(model)
          attrs, files = visualization_inner(vi, model, checkBoxes, False)
          for a, f in zip(attrs, files):
            a.save(UPLOAD_FOLDER + '/results/' + f[:-4] + '_guidedbackprop_vgg.jpg')
        
      elif viz == 'saliency':
          vi = Saliency(model)
          attrs, files = visualization_inner(vi, model, checkBoxes, False)
          for a, f in zip(attrs, files):
            a.save(UPLOAD_FOLDER + '/results/' + f[:-4] + '_saliency_vgg.jpg')

      elif viz == 'occlusion':
          vi = Occlusion(model)
          attrs, files = visualization_inner(vi, model, checkBoxes, True)
          for a, f in zip(attrs, files):
            a.save(UPLOAD_FOLDER + '/results/' + f[:-4] + '_occlusion_vgg.jpg')


    if net == 'resnet':
      model = models.resnet50(pretrained=True)
      model.eval()
      model = model.cuda()
      model = model.float()
      if viz == 'guidedbackprop':
          replace_layers(model, nn.ReLU, nn.ReLU(inplace=False))
          vi = GuidedBackprop(model)
          attrs, files = visualization_inner(vi, model, checkBoxes, False)
          for a, f in zip(attrs, files):
            a.save(UPLOAD_FOLDER + '/results/' + f[:-4] + '_guidedbackprop_resnet.jpg')
        
      elif viz == 'saliency':
          vi = Saliency(model)
          attrs, files = visualization_inner(vi, model, checkBoxes, False)
          for a, f in zip(attrs, files):
            a.save(UPLOAD_FOLDER + '/results/' + f[:-4] + '_saliency_resnet.jpg')
        
      elif viz == 'occlusion':
          vi = Occlusion(model)
          attrs, files = visualization_inner(vi, model, checkBoxes, True)
          for a, f in zip(attrs, files):
            a.save(UPLOAD_FOLDER + '/results/' + f[:-4] + '_occlusion_resnet.jpg')