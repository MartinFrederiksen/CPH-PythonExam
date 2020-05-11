import libs.FacialRecognitionData as frd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import imutils
import zipfile
import tarfile
import shutil
import cv2
import os

def filename_check(folder, filename):
    original_fname = filename
    counter = 0
    while os.path.isfile(os.path.join(folder, filename)):
        fname, extension = original_fname.split('.')
        filename = fname + '({0}).'.format(counter) + extension
        counter += 1
    return filename

def Unzip_File(path, dist):
    print('== Unzipping File ==')
    filename = os.path.basename(path).split('.')[0]

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zipObj:
            dist = os.path.join(dist, filename)
            zipObj.extractall(dist)
    
    return dist

def zip_directory(path, dist):
    print('== Zipping File ==')
    filename = os.path.basename(path)
    filename = filename + "_sorted"
    dist = os.path.join(dist, filename)

    shutil.make_archive(dist, 'zip', path)
    return filename + '.zip'

def train_faces(train_folder):
    print('== Started training of Faces ==')
    pool = Pool(cpu_count())
    faces = _tuple_array_to_dictionary(list(tqdm(pool.imap(_train_faces_iteration, [(f, train_folder) for f in os.listdir(train_folder)]), total=len(os.listdir(train_folder)))))
    pool.close()
    pool.join()
    return faces

def _train_faces_iteration(args):
    filename, train_folder = args
    fname = filename.split('.')[0]
    image = cv2.imread(os.path.join(train_folder, filename))
    image = imutils.resize(image, width=500)
    return (fname, frd.face_encodings_data(image)[0])

def create_face_folders(dist, train_folder, faces):
    print('== Started creation of Face Folders ==')
    for face in tqdm(faces):
        os.makedirs(os.path.join(dist, face))
    os.makedirs(os.path.join(dist, "Unknown"))

    for image in os.listdir(train_folder):
        shutil.copy2(os.path.join(train_folder, image), dist)

    shutil.rmtree(os.path.join(dist, "train"), ignore_errors=True)

def sort_images(unzipped_dist, known_name_images):
    print('== Started Sorting of Images ==')
    files = [f for f in os.listdir(unzipped_dist) if os.path.isfile(os.path.join(unzipped_dist, f))]

    pool = Pool(cpu_count())
    sorted_images = _tuple_array_to_dictionary_array(list(tqdm(pool.imap(_sort_images_itteration, [(f, unzipped_dist, known_name_images) for f in files]), total=len(files))))
    pool.close()
    pool.join()

    _copy_files_from_root(unzipped_dist, sorted_images)

def _sort_images_itteration(args):
    filename, unzipped_dist, known_name_images = args

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(os.path.join(unzipped_dist, filename))
        image = imutils.resize(image, width=300)
        face_locations = frd.face_location_data(image)
        face_encodings = frd.face_encodings_data(image, face_locations)
        
        faces = []
        for face_encoding in face_encodings:
            face_matches = frd.compare_faces(known_name_images, face_encoding, 0.65)
            face_name = "Unknown"

            face_distances = frd.face_distance(known_name_images, face_encoding)
            face_match_index = np.argmin(face_distances)
            if face_matches[face_match_index]:
                face_name = list(known_name_images.keys())[face_match_index]
            
            faces.append((face_name, filename))
        return faces

def _copy_files_from_root(unzipped_dist, sorted_images):
    for name in sorted_images:
        for image in sorted_images[name]:
            shutil.copy2(os.path.join(unzipped_dist, image), os.path.join(unzipped_dist, name))

    for filename in os.listdir(unzipped_dist):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(unzipped_dist, filename))

def _tuple_array_to_dictionary_array(tuple_array):
    sorted_images = {}
    for face in tuple_array:
        if face is not None:
            for face_name, filename in face:
                if face_name not in sorted_images:
                    sorted_images[face_name] = []
                sorted_images[face_name].append(filename)
    return sorted_images

def _tuple_array_to_dictionary(tuple_array):
    di = {}
    for a, b in tuple_array:
        di[a] = b
    return di