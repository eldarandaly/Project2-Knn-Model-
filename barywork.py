# importing required libraries
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import sys
import time
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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

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
        #draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

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


# Main window class
class MainWindow(QMainWindow):

	# constructor
	def __init__(self):
		super().__init__()
	
		# setting geometry
		self.setGeometry(100,100,800, 600)

		# setting style sheet
		self.setStyleSheet("background : lightgrey;")

		# getting available cameras
		self.available_cameras = QCameraInfo.availableCameras()

		# if no camera found
		if not self.available_cameras:
			# exit the code
			sys.exit()

		# creating a status bar
		self.status = QStatusBar()

		# setting style sheet to the status bar
		self.status.setStyleSheet("background : white;")

		# adding status bar to the main window
		self.setStatusBar(self.status)

		# path to save
		self.save_path = ""

		# creating a QCameraViewfinder object
		self.viewfinder = QCameraViewfinder()

		# showing this viewfinder
		self.viewfinder.show()

		# making it central widget of main window
		self.setCentralWidget(self.viewfinder)

		# Set the default camera.
		self.select_camera('rtsp://admin:TZZUNI@192.168.1.58/')

		# creating a tool bar
		toolbar = QToolBar("Camera Tool Bar")

		# adding tool bar to main window
		self.addToolBar(toolbar)

		# creating a photo action to take photo
		click_action = QAction("Click photo", self)

		# adding status tip to the photo action
		click_action.setStatusTip("This will capture picture")

		# adding tool tip
		click_action.setToolTip("Capture picture")


		# adding action to it
		# calling take_photo method
		click_action.triggered.connect(self.click_photo)

		# adding this to the tool bar
		toolbar.addAction(click_action)

		# similarly creating action for changing save folder
		change_folder_action = QAction("Change save location",
									self)

		MyDetect=QAction("Start",self)
		MyDetect.triggered.connect(self.Recog)
		toolbar.addAction(MyDetect)

		# adding status tip
		change_folder_action.setStatusTip("Change folder where picture will be saved saved.")

		# adding tool tip to it
		change_folder_action.setToolTip("Change save location")

		# setting calling method to the change folder action
		# when triggered signal is emitted
		change_folder_action.triggered.connect(self.change_folder)

		# adding this to the tool bar
		toolbar.addAction(change_folder_action)


		# creating a combo box for selecting camera
		camera_selector = QComboBox()

		# adding status tip to it
		camera_selector.setStatusTip("Choose camera to take pictures")

		# adding tool tip to it
		camera_selector.setToolTip("Select Camera")
		camera_selector.setToolTipDuration(2500)

		# adding items to the combo box
		camera_selector.addItems([camera.description()
								for camera in self.available_cameras])

		# adding action to the combo box
		# calling the select camera method
		camera_selector.currentIndexChanged.connect(self.select_camera)

		# adding this to tool bar
		toolbar.addWidget(camera_selector)

		# setting tool bar stylesheet
		toolbar.setStyleSheet("background : white;")



		# setting window title
		self.setWindowTitle("PyQt5 Cam")

		# showing the main window
		self.show()

	# method to select camera
	def select_camera(self, i):

		# getting the selected camera
		self.camera = QCamera(self.available_cameras[i])

		# setting view finder to the camera
		self.camera.setViewfinder(self.viewfinder)

		# setting capture mode to the camera
		self.camera.setCaptureMode(QCamera.CaptureStillImage)

		# if any error occur show the alert
		self.camera.error.connect(lambda: self.alert(self.camera.errorString()))

		# start the camera
		self.camera.start()

		# creating a QCameraImageCapture object
		self.capture = QCameraImageCapture(self.camera)

		# showing alert if error occur
		self.capture.error.connect(lambda error_msg, error,
								msg: self.alert(msg))

		# when image captured showing message
		self.capture.imageCaptured.connect(lambda d,
										i: self.status.showMessage("Image captured : "
																	+ str(self.save_seq)))

		# getting current camera name
		self.current_camera_name = self.available_cameras[i].description()

		# initial save sequence
		self.save_seq = 0

	# method to take photo
	def click_photo(self):

		# time stamp
		timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")

		# capture the image and save it on the save path
		self.capture.capture(os.path.join(self.save_path,
										"%s-%04d-%s.jpg" % (
			self.current_camera_name,
			self.save_seq,
			timestamp
		)))

		# increment the sequence
		self.save_seq += 1

	# change folder method
	def change_folder(self):

		# open the dialog to select path
		path = QFileDialog.getExistingDirectory(self,
												"Picture Location", "")

		# if path is selected
		if path:

			# update the path
			self.save_path = path

			# update the sequence
			self.save_seq = 0

	# method for alerts
	def alert(self, msg):

		# error message
		error = QErrorMessage(self)

		# setting text to the error message
		error.showMessage(msg)

	def Recog (self):
		cap=cv2.VideoCapture(1)
		# creating a QCameraViewfinder object
		#self.viewfinder = QCameraViewfinder()

		# showing this viewfinder
		#self.viewfinder.show()

		# making it central widget of main window
		#self.setCentralWidget(self.viewfinder)
		while True:
			ret,frame=cap.read()
			
			predictions=predict(frame,model_path="trained_knn_modelOneShot1.clf")
			frame = show_prediction_labels_on_image(frame, predictions)
			cv2.imshow('camera', frame)


			# showing this viewfinder
			#making it central widget of main window
			if ord('q') == cv2.waitKey(1):
				cap.release()
				cv2.destroyAllWindows()
				exit(0)

# Driver code
if __name__ == "__main__" :
	
# create pyqt5 app
    App = QApplication(sys.argv)

# create the instance of our Window
    window = MainWindow()

# start the app
    sys.exit(App.exec())
