import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
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
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
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


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        process_this_frame=9
        while self.ThreadActive:
            try:
                ret, frame = Capture.read()
                if ret:
                    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,5)
                    for (x,y,w,h)in faces:
                        face=Image[y-5:y+h+5,x-5:x+w+5]
                        resized_face=cv2.resize(face,(160,160))
                        resized_face = resized_face.astype("float") / 255.0
                        resized_face = np.expand_dims(resized_face, axis=0)
                        preds = model.predict(resized_face)[0]
                        if preds>0.4:
                            label = 'spoof'
                            cv2.putText(Image, label, (x,y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                            cv2.rectangle(Image, (x, y), (x+w,y+h),
                                (0, 0, 255), 2)
                        else:
                            label = 'real'
                            cv2.putText(Image, label, (x,y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            cv2.rectangle(Image, (x, y), (x+w,y+h),
                                (0, 255, 0), 2)
                                #Start Here -------------------------------- To Be Fixed
                    process_this_frame = process_this_frame + 1
                    if process_this_frame % 10 == 0:
                        predictions = predict(Image, model_path="trained_knn_modelOneShot1.clf") 
                    Image = show_prediction_labels_on_image(Image, predictions)
                    #cv2.putText(frame,f'FPS:{int(fps)}',(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                    #FlippedImage = cv2.flip(Image,1)
                    ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)
            except Exception as e:
                pass        

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())