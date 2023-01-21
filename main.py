from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from utils import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # run the image through your machine learning model here
        # ...
        output = predict(filepath)
        return render_template('result.html', output=output, image_file=filepath)
    else:
        return 'Invalid file format. Please upload an image file.'

if __name__ == '__main__':
    app.run(debug=True)
