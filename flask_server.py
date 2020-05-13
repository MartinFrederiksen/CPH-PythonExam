from flask import Flask, render_template, request, send_from_directory, abort
from werkzeug.utils import secure_filename
from config import config
from tqdm import tqdm
import file_handeling
import time
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/uploader', methods=['POST'])
def upload_file():
    start_time = time.time()

    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filename = file_handeling.filename_check(config['UPLOAD_FOLDER'], filename)

        path = os.path.join(config['UPLOAD_FOLDER'], filename)
        f.save(path)

        sorted_zip = file_handeling.handle_zip_file(path)

        try:
            end_time = time.time()
            print('== Time Elapsed: %.2f seconds ==' % (end_time - start_time))
            return send_from_directory(config['ZIP_FOLDER'], filename=sorted_zip, as_attachment=True)
        except FileNotFoundError:
            abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = True)