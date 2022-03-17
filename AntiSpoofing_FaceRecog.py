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
import time
import imutils
from threading import Thread
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)






def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    IDs=[]
    TestID=[]
    path=train_dir
    #cl='images/train'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)       
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

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
    return [(pred, loc) if rec else ("unknown.NoID", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.

    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        ID=name.split('.')[1]
        name=name.split('.')[0]
        #print(name,ID)
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        draw.text((left + 6, bottom - text_height + 5), "ID:"+ID, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":

    x=input("enter t for training or s to start\n")
    if x=='t':
        print("Training KNN classifier...")
        classifier = train("images/train", model_save_path="trained_knn_modelOneShot1.clf", n_neighbors=1)
        print("Training complete!")
    elif x=='s':
        # process one frame in every 30 frames for speed
        process_this_frame = 29
        sT=0
        print('Setting cameras up...')
        # muliple cameras can be used with the format url = 'http://username:password@camera_ip:port'
        url = 0#'./test.mp4'
        cap = cv2.VideoCapture(url)
        while True:
            try:
                ret, frame = cap.read()
            
                ctime=time.time()
                fps=1/(ctime-sT)
                sT=ctime


                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:  
                    face = frame[y-5:y+h+5,x-5:x+w+5]
                    resized_face = cv2.resize(face,(160,160))
                    resized_face = resized_face.astype("float") / 255.0
                    # resized_face = img_to_array(resized_face)
                    resized_face = np.expand_dims(resized_face, axis=0)
                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"
                    #print(preds)

                   # process_this_frame = process_this_frame + 1
                    #if process_this_frame % 30 == 0:
                    predictions = predict(frame, model_path="trained_knn_modelOneShot1.clf") # low fps from predictio 
                    preds = model.predict(resized_face)[0]

                    if preds> 0.5:
                    
                        label = 'spoof'
                        cv2.putText(frame, label, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)       
                        frame = show_prediction_labels_on_image(frame, predictions)
                        cv2.putText(frame,f'FPS:{int(fps)}',(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                    
                    else:

                        label = 'real'
                        cv2.putText(frame, label, (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        frame = show_prediction_labels_on_image(frame, predictions)
                        cv2.putText(frame,f'FPS:{int(fps)}',(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                
                cv2.imshow('camera', frame)
                if ord('q') == cv2.waitKey(1):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
            except Exception as e:
                pass
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming

            #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
               # process_this_frame = process_this_frame + 1
                #if process_this_frame % 40 == 0:
                   # predictions = predict(frame, model_path="trained_knn_modelOneShot1.clf") # low fps from predictio 
               # frame = show_prediction_labels_on_image(frame, predictions)
              #  cv2.putText(frame,f'FPS:{int(fps)}',(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

