# This is update on KNN Model With AntiSpoofing And PyQt5

## TO Run the Python file open PyQtTest.py 

## this is update on KNN Model With AntiSpoofing And PyQt5

# Run PyQT Test .py and change From
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        video_capture = cv2.VideoCapture("rtsp://admin:TZZUNI@192.168.1.58:554/H.264", cv2.CAP_FFMPEG)
        video_capture.set(cv2.CAP_PROP_FPS, 60) 
        fresh = FreshestFrame(video_capture) 
        
To 
        
        #os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FPS, 60) 
        fresh = FreshestFrame(video_capture) 

