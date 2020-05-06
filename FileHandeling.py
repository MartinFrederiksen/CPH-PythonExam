import libs.FacialRecognitionData as frd
import zipfile
import tarfile
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import os

def CheckExists(folder, filename):
    original_fname = filename
    counter = 0
    while os.path.isfile(os.path.join(folder, filename)):
        fname, extension = original_fname.split('.')
        filename = fname + '({0}).'.format(counter) + extension
        counter += 1
    return filename

def Unzip_File(path, dist):
    filename = os.path.basename(path).split('.')[0]

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zipObj:
            dist = os.path.join(dist, filename)
            zipObj.extractall(dist)

    faces = train_faces(os.path.join(dist, "train"))
    for face in faces:
        os.makedirs(os.path.join(dist, face))
    os.makedirs(os.path.join(dist, "Unknown"))

    si = sort_images(dist, faces)
    for name in si:
        for image in si[name]:
            shutil.copy2(os.path.join(dist, image), os.path.join(dist, name))

    for filename in os.listdir(dist):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(dist, filename))

    shutil.rmtree(os.path.join(dist, "train"), ignore_errors=True)

    return dist

def Zip_Dir(path, dist):
    filename = os.path.basename(path)
    filename = filename + "_sorted"
    dist = os.path.join(dist, filename)

    shutil.make_archive(dist, 'zip', path)
    return filename + '.zip'

def train_faces(train_folder):
    known_name_images = {}
    for filename in os.listdir(train_folder):
        fname = filename.split('.')[0]
        known_name_images[fname] = frd.face_encodings_data(cv2.imread(os.path.join(train_folder, filename)))[0]
    return known_name_images

def sort_images(sort_folder, known_name_images):
    sorted_images = {}

    for filename in tqdm(os.listdir(sort_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(os.path.join(sort_folder, filename))
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            face_locations = frd.face_location_data(image)
            face_encodings = frd.face_encodings_data(image, face_locations)
            
            for face_encoding in face_encodings:
                face_matches = frd.compare_faces(known_name_images, face_encoding, 0.68)
                face_name = "Unknown"

                face_distances = frd.face_distance(known_name_images, face_encoding)
                face_match_index = np.argmin(face_distances)
                if face_matches[face_match_index]:
                    face_name = list(known_name_images.keys())[face_match_index]
                
                if face_name not in sorted_images:
                    sorted_images[face_name] = []
                sorted_images[face_name].append(filename)

    return sorted_images

