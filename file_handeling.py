from tensorflow.keras.models import load_model
from multiprocessing import Pool, cpu_count
import libs.FacialRecognitionData as frd
from keras.preprocessing import image
from config import config
from tqdm import tqdm
import numpy as np
import argparse
import imutils
import zipfile
import tarfile
import shutil
import time
import cv2
import os

# Names of the objects that can be sorted by
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def handle_zip_file(zip_file_path, tolerance=0.63, model='v1'):
    """
        Handles the ZIP file, by calling all the necessary functions

        :param zip_file_path: Path to the uploaded ZIP File
        :param tolerance: Tolerance of the facial recognition (Default = 0.63)
        :param model: model version to be used for recognition (Default = v1)
    """
    # Start timer
    start_time = time.time()

    # Unzip the file to Unzip Folder
    unzipped_dist = unzip_file(zip_file_path, config['UNZIP_FOLDER'])

    # Train the Facial Recognition on the faces in the train folder
    train_folder = os.path.join(unzipped_dist, "train")
    faces = train_faces(train_folder)

    # Moves images to root and removes folder
    clean_up_train_folder(unzipped_dist, train_folder)

    # Sort images to their dedicated folders depending on face on the image
    sort_faces(unzipped_dist, faces, tolerance)

    # Sort objects in unknown folder if folder exsists
    sort_unknown(unzipped_dist, model)

    # Zip the file again
    zipped_file = zip_directory(unzipped_dist, config['ZIP_FOLDER'])
    
    # End timer and print time in seconds
    end_time = time.time()
    print('== Time Elapsed: %.2f seconds ==' % (end_time - start_time))

    return (zipped_file, unzipped_dist)

def filename_check(folder, filename):
    """
        Checks the filename for duplicates, and adds count after if colliding

        :param path: Location of zipped file
        :param dist: Destination the zip file shall be unzipped to
    """

    original_fname = filename
    counter = 0
    while os.path.isfile(os.path.join(folder, filename)):
        fname, extension = original_fname.split('.')
        filename = fname + '({0}).'.format(counter) + extension
        counter += 1
    return filename

def unzip_file(path, dist):
    """
        Unzippes the ZIP file at path to the dist

        :param path: Location of zipped file
        :param dist: Destination the ZIP file shall be unzipped to
    """

    print('== Unzipping File ==')
    filename = os.path.basename(path).split('.')[0]

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zipObj:
            dist = os.path.join(dist, filename)
            zipObj.extractall(dist)
    
    return dist

def zip_directory(path, dist):
    """
        ZIPs the directory at path and saves it to dist

        :param path: Location of folder to be zipped
        :param dist: Destination the ZIP file shall be saved to
    """

    print('== Zipping File ==')
    filename = os.path.basename(path)
    filename = filename + "_sorted"
    dist = os.path.join(dist, filename)

    shutil.make_archive(dist, 'zip', path)
    return filename + '.zip'

def train_faces(train_folder):
    """
        Trains faces of the trainfolder and returns a dictionary of the faces encodings

        :param train_folder: Location of folder of images to train on
    """

    print('== Started training of Faces ==')
    pool = Pool(cpu_count())
    faces = _tuple_array_to_dictionary_single(
        list(
            tqdm(
                pool.imap(_train_faces_iteration, [(f, train_folder) for f in os.listdir(train_folder)]), 
                total=len(os.listdir(train_folder)))))
    pool.close()
    pool.join()
    return faces

def _train_faces_iteration(args):
    filename, train_folder = args
    """
        Iteration of face training, takes image finds face and returns face encoding

        :param filename: Name of the image file to be trained
        :param train_folder: Folder where the image file is located
    """

    fname = filename.split('.')[0]
    image = cv2.imread(os.path.join(train_folder, filename))
    image = imutils.resize(image, width=500)
    return (fname, frd.face_encodings_data(image)[0])
    
def clean_up_train_folder(dist, train_folder):
    """
        Cleans up the train folder after training, moves the folders to root and removes the folder

        :param dist: Destination to root folder
        :param train_folder: Folder where the image file is located
    """

    for image in os.listdir(train_folder):
        shutil.move(os.path.join(train_folder, image), dist)

    shutil.rmtree(os.path.join(dist, "train"), ignore_errors=True)


def sort_faces(unzipped_dist, known_name_images, tolerance):
    """
        Sorts the images inside the root folder based on the faces on the images

        :param unzipped_dist: Destination of the unzipped folder
        :param known_name_images: Dictionary of known faces from training
        :param tolerance: Tolerance of the facial recognition (Default = 0.63)
    """

    print('== Started Sorting of Images ==')
    files = [f for f in os.listdir(unzipped_dist) if os.path.isfile(os.path.join(unzipped_dist, f))]

    pool = Pool(cpu_count())
    sorted_images = _face_tuple_array_to_dictionary(
        list(
            tqdm(
                pool.imap(_sort_faces_iteration, [(f, unzipped_dist, known_name_images, tolerance) for f in files]), total=len(files))))
    pool.close()
    pool.join()

    # Create folders for each learned face
    create_folders(unzipped_dist, sorted_images)
    copy_files_to_folder(unzipped_dist, unzipped_dist, sorted_images)

def _sort_faces_iteration(args):
    filename, unzipped_dist, known_name_images, tolerance = args
    """
        Iteration of face sorting, takes image finds face and returns name of the face

        :param filename: Name of the image file to be sorted
        :param unzipped_dist: Destination of the unzipped folder
        :param known_name_images: Dictionary of known faces from training
        :param tolerance: Tolerance of the facial recognition (Default = 0.63)
    """

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(os.path.join(unzipped_dist, filename))
        image = imutils.resize(image, width=300)

        face_locations = frd.face_location_data(image)
        face_encodings = frd.face_encodings_data(image, face_locations)
        
        faces = []
        face_names = []
        for face_encoding in face_encodings:
            face_matches = frd.compare_faces(known_name_images, face_encoding, tolerance)

            face_distances = frd.face_distance(known_name_images, face_encoding)
            face_match_index = np.argmin(face_distances)
            if face_matches[face_match_index]:
                face_names.append(list(known_name_images.keys())[face_match_index])
        
        faces.append(('-'.join(face_names) if len(face_names) > 0 else "Unknown", filename))
        return faces

def sort_unknown(unzipped_dist, model):
    """
        Sorts the images inside the Unkown folder based on the objects on the images

        :param unzipped_dist: Destination of the unzipped folder
        :param model: model version to be used for recognition (Default = v1)
    """

    if os.path.isdir(os.path.join(unzipped_dist, 'Unknown')):
        files = os.listdir(os.path.join(unzipped_dist, 'Unknown'))

        pool = Pool(cpu_count())
        sorted_images = _tuple_array_to_dictionary_array(
            list(
                tqdm(
                    pool.imap(_sort_unknown_iteration, [(fname, unzipped_dist, model) for fname in files]), total=len(files))))
        pool.close()
        pool.join()

        create_folders(unzipped_dist, sorted_images)
        copy_files_to_folder(unzipped_dist, os.path.join(unzipped_dist, "Unknown"), sorted_images)

def _sort_unknown_iteration(args):
    filename, unzipped_dist, model  = args
    """
        Iteration of object sorting, takes image finds objects and returns name of the object

        :param filename: Name of the image file to be sorted
        :param unzipped_dist: Destination of the unzipped folder
        :param model: model version to be used for recognition (Default = v1)
    """

    loaded_model = load_model('training/cifar10_model_' + model + '.h5')
    img = image.load_img(os.path.join(os.path.join(unzipped_dist, "Unknown"), filename), target_size=(32, 32))
    img = np.expand_dims(img, axis=0)
    result = loaded_model.predict_classes(img)
    return (class_names[result[0]], filename)


def create_folders(dist, names):
    """
        Creates the folders in names at the dist

        :param dist: Destination where to create the folders
        :param names: Array or Dictionary of folders names to create
    """

    print('== Started creation of Folders ==')
    for name in tqdm(names):
        os.makedirs(os.path.join(dist, name))

def copy_files_to_folder(dist, image_location, sorted_images):
    """
        Moves the images from image_location to the folder the image is connected to

        :param dist: Destination where the named folder is located
        :param image_location: Location of the images to move
        :param sorted_images: Dictionary of folder names with arrays of images to go there 
    """

    print('== Started moving of images ==')
    for name in tqdm(sorted_images):
        for image in sorted_images[name]:
            if os.path.exists(os.path.join(image_location, image)):
                shutil.move(os.path.join(image_location, image), os.path.join(dist, name))


def _face_tuple_array_to_dictionary(tuple_array):
    sorted_images = {}
    for face in tuple_array:
        if face is not None:
            for face_name, filename in face:
                if face_name not in sorted_images:
                    sorted_images[face_name] = []
                sorted_images[face_name].append(filename)
    return sorted_images

def _tuple_array_to_dictionary_single(tuple_array):
    di = {}
    for a, b in tuple_array:
        di[a] = b
    return di

def _tuple_array_to_dictionary_array(tuple_array):
    di = {}
    for a, b in tuple_array:
        if a not in di:
            di[a] = []
        di[a].append(b)
    return di


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program that takes a zip file and sorts it by faces on images')
    parser.add_argument('zip_file', help='a path to the zip file')
    parser.add_argument('-t', '--tolerance', default=0.63, help='an integer for the tolerance of distance between face matches')
    args = parser.parse_args()

    handle_zip_file(args.zip_file, args.tolerance)