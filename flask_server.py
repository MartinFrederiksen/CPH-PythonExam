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

        # Unzip the file to Unzip Folder
        unzipped_dist = file_handeling.Unzip_File(path, config['UNZIP_FOLDER'])

        # Train the Facial Recognition on the faces in the train folder
        train_folder = os.path.join(unzipped_dist, "train")
        faces = file_handeling.train_faces(train_folder)

        # Create folders for each learned face
        file_handeling.create_face_folders(unzipped_dist, train_folder, faces)

        # Sort images to their dedicated folders depending on face on the image
        file_handeling.sort_images(unzipped_dist, faces)

        # Zip the file again
        sorted_zip = file_handeling.zip_directory(unzipped_dist, config['ZIP_FOLDER'])
        try:
            end_time = time.time()
            print('== Time Elapsed: %.2f seconds ==' % (end_time - start_time))
            return send_from_directory(config['ZIP_FOLDER'], filename=sorted_zip, as_attachment=True)
        except FileNotFoundError:
            abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = True)