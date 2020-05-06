import numpy as np
import dlib
import sys

face_detector = dlib.get_frontal_face_detector()


"""
    shape_predictor_68_face_landmarks.dat is trained on the ibug 300-W dataset and gives back 68 face landmarks

    dlib_face_recognition_resnet_model_v1.dat is a pre learned dataset of about 3 million faces.
    The model is a ResNet network with 29 conv layers.

    More can be read at https://github.com/davisking/dlib-models
"""
predictor_68_point_model = "libs/models/shape_predictor_68_face_landmarks.dat"
face_recognition_model = "libs/models/dlib_face_recognition_resnet_model_v1.dat"
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _rect_to_cords(rect):
    """
        Converts a dlib rect to cordinates (t, r, b, l)

        :param rect: A dlib rectangle object
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _cords_to_rect(cords):
    """
        Converts cordinates (t, r, b, l) to a dlib rect 

        :param cords: A tuple of cords (t, r, b, l)
    """
    return dlib.rectangle(cords[3], cords[0], cords[1], cords[2])


def face_distance(face_encodings, face_to_compare):
    """
        Retuns the euclidean distance to a known face if any, given the face_encodings of known faces

        :param face_encodings: List of face encodings
        :face_to_compare: A face encoding to compare
    """

    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(list(face_encodings.values()) - face_to_compare, axis=1)

def face_location_data(image):
    """
        Returns an array of bounding boxes of human faces in a image

        :param image: Image to search in (as numpy array)
    """
    return [_rect_to_cords(face) for face in face_detector(image, 1)]

def _faces_landmarks_data(image, face_locations=None):
    """
        Returns raw landmarks of all faces in image (68 points pr. face)

        :param image: image to find face landmarks in
        :param face_locations: Optionally faces locations to get landmarks from
    """
    if face_locations == None:
        face_locations = [_cords_to_rect(location) for location in face_location_data(image)]
    else:
        face_locations = [_cords_to_rect(location) for location in face_locations]

    return [pose_predictor_68_point(image, face_location) for face_location in face_locations]

def _faces_landmarks_data(image, face_locations=None):
    """
        Returns raw landmarks of all faces in image (68 points pr. face)

        :param image: image to find face landmarks in
        :param face_locations: Optionally faces locations to get landmarks from
    """
    if face_locations == None:
        face_locations = [_cords_to_rect(location) for location in face_location_data(image)]
    else:
        face_locations = [_cords_to_rect(location) for location in face_locations]

    return [pose_predictor_68_point(image, face_location) for face_location in face_locations]

def faces_landmarks_dict(image, face_locations=None):
    """
        Returns a dictionary of all face featues given an image of all faces in image

        :param image: image to find face landmarks in
        :param face_locations: Optionally faces locations to get landmarks from
    """

    landmarks = _faces_landmarks_data(image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
    
    return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks_as_tuples]

def face_encodings_data(image, face_locations=None):
    """
        Returns an 128-dimension face encodings given an image

        :param image: image to find face landmarks in
        :param face_locations: Optionally faces locations to get landmarks from
    """

    landmarks = _faces_landmarks_data(image, face_locations)
    return [np.array(face_encoder.compute_face_descriptor(image, raw_landmark_set, 1)) for raw_landmark_set in landmarks]

def compare_faces(known_face_encodings, face_encoding, tolerance=0.63):
    """
        Retuns a list of faces the encoding have matched againts

        :param known_face_encodings: a lsit of known face encodings
        :param face_encodings: a single face encoding to check against the list
        :param tolerance: How much distance between face encodings to consider it a match
    """

    return list(face_distance(known_face_encodings, face_encoding) <= tolerance)