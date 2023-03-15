from flask import Flask, request



app = Flask(__name__)  # This is needed so that Flask knows where to look for resources such as templates and static files.

@app.route("/")  # By default, a route only answers to GET requests.
def hello_world():
    return "<p>Hello, World!</p>"  # The default content type is HTML, so HTML in the string will be rendered by the browser.

from markupsafe import escape

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f'Post {post_id}'

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return f'Subpath {escape(subpath)}'

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')         # Accessing the URL with a trailing slash (/about/) produces a 404 “Not Found” error. 
def about():                 # This helps keep URLs unique for these resources, which helps search engines avoid 
    return 'The about page'  # indexing the same page twice.

from flask import url_for

@app.route('/login')
def login():
    return 'login'

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'

with app.test_request_context():
    print(url_for('hello_world'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))


@app.route('/login', methods=['GET', 'POST'])
def login_():                      # The example above keeps all methods for the route within one function,
    if request.method == 'POST':   # which can be useful if each part uses some common data.
        return do_the_login()
    else:
        return show_the_login_form()

@app.get('/login')
def login_get():
    return show_the_login_form()

@app.post('/login')
def login_post():
    return do_the_login()

# To generate URLs for static files, use the special 'static' endpoint name:
# url_for('static', filename='style.css')  # The file has to be stored on the filesystem as static/style.css.

from flask import render_template

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

# a module:
# /application.py
# /templates
#    /hello.html

"""
@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error=error)
"""

# To access parameters submitted in the URL (?key=value) you can use the args attribute:

#searchword = request.args.get('key', '')

from werkzeug.utils import secure_filename

"""
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['the_file']
        file.save(f"/var/www/uploads/{secure_filename(file.filename)}")


Reading cookies:

from flask import request

@app.route('/')
def index():
    username = request.cookies.get('username')
    # use cookies.get(key) instead of cookies[key] to not get a
    # KeyError if the cookie is missing.

Storing cookies:

from flask import make_response

@app.route('/')
def index():
    resp = make_response(render_template(...))
    resp.set_cookie('username', 'the username')
    return resp


"""