from flask import Flask, render_template, request, send_from_directory, abort, jsonify
from werkzeug.utils import secure_filename
from config import config
from tqdm import tqdm
import file_handeling
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/zip/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    tolerance = float(request.form['tolerance'])
    model = request.form['model']

    filename = secure_filename(f.filename)
    filename = file_handeling.filename_check(config['UPLOAD_FOLDER'], filename)

    path = os.path.join(config['UPLOAD_FOLDER'], filename)
    f.save(path)

    zipped_file, unzipped_dist = file_handeling.handle_zip_file(path, tolerance=tolerance, model=model)

    information = {
        "originalName" : path.split('/')[-1],
        "staticName" : unzipped_dist.split('/')[-1],
        "zippedName" : zipped_file.split('/')[-1],
        "staticFolders" : {
            key: os.listdir(os.path.join(unzipped_dist, key)) for key in os.listdir(unzipped_dist)
        }
    }

    try:
        return jsonify(information)
    except FileNotFoundError:
        abort(404)

@app.route('/zip/download', methods=['GET'])
def download_zip():
    zip_type = request.args['type']
    zip_file = request.args['zip']

    try:
        return send_from_directory(config['ZIP_FOLDER'] if zip_type == 'sorted' else config['UPLOAD_FOLDER'], filename=zip_file, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = False)