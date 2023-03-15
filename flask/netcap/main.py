from flask import Flask, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter



UPLOAD_FOLDER = '/home/cosmos/Documents/Miletos/Missing Semester/git/git_exp/flask/netcap/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)  # This is needed so that Flask knows where to look for resources such as templates and static files.
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            image = Image.open(UPLOAD_FOLDER + '/' + file.filename)
            image = image.convert('L')
            image = image.filter(ImageFilter.FIND_EDGES)
            image.save(UPLOAD_FOLDER + '/edges/' + file.filename[:-4] + '_edge.jpg')
            return redirect(url_for('download_file', name=file.filename[:-4] + '_edge.jpg'))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/edges/<name>')
def download_file(name):
    return send_from_directory(UPLOAD_FOLDER + '/edges/', name)

#@app.route('/uploads/edges', methods=['GET'])
#def show_edges():