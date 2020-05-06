from flask import Flask, render_template, request, send_from_directory, abort
from werkzeug.utils import secure_filename
from tqdm import tqdm
import os
import FileHandeling
from Config import config

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filename = FileHandeling.CheckExists(config['UPLOAD_FOLDER'], filename)

        path = os.path.join(config['UPLOAD_FOLDER'], filename)
        f.save(path)

        unzipped = FileHandeling.Unzip_File(path, config['UNZIP_FOLDER'])

        try:
            sorted_zip = FileHandeling.Zip_Dir(unzipped, config['ZIP_FOLDER'])
            return send_from_directory(config['ZIP_FOLDER'], filename=sorted_zip, as_attachment=True)
        except FileNotFoundError:
            abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = True)