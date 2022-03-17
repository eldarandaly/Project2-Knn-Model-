import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



def show_prediction_labels_on_image():

    #pil_image = Image.fromarray(frame)
    #Draw = ImageDraw.Draw(pil_image)
    url = './ll.mp4'
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        predictions = predict(frame, model_path="trained_knn_modelOneShot.clf")

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        if True:    
            for name, (top, right, bottom, left) in predictions:
                # enlarge the predictions for the full sized image.
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1
                # Draw a box around the face using the Pillow module
               # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                predictions = predict(frame, model_path="trained_knn_modelOneShot.clf")

                # There's a bug in Pillow where it blows up with non-UTF-8 text
                # when using the default bitmap font
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                print(name)
        # Remove the drawing library from memory as per the Pillow docs.
        #del draw
        # Save image in open-cv format to be able to show it.
                cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

if __name__ == "__main__":
    print('Setting cameras up...')
    while True:
        show_prediction_labels_on_image()

'''
if __name__ == "__main__":
    print("Training KNN classifier...")

    #classifier = train("images/train", model_save_path="trained_knn_modelOneShot.clf", n_neighbors=2)
    print("Training complete!")
    # process one frame in every 30 frames for speed
    process_this_frame = 4
    print('Setting cameras up...')
    # muliple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = './ll.mp4'
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            #process_this_frame = process_this_frame + 1
            #if process_this_frame % 5 == 0:
            predictions = predict(frame, model_path="trained_knn_modelOneShot.clf")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
'''