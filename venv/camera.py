import sys
print(sys.path)
sys.path.clear()
sys.path.append('/Users/sharad/Downloads/HackDukeML/venv')
sys.path.append('/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/lib/python3.8')
sys.path.append('/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/lib/python3.8/lib-dynload')
sys.path.append('/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/lib/python3.8/site-packages')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages')
sys.path.append('/Users/sharad/Library/Python/3.8/lib/python/site-packages')
print(sys.path)



import cv2
from model import FacialExpressionModel
import numpy as np

#facec = cv2.CascadeClassifier('/Users/sharad/Library/Python/3.8/lib/python/site-packages/cv2/data/haarcascade_frontalface_default.xml')
facec = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#CHANGE #5: the inputs to the model (saved and format) (same as Change #3)
model = FacialExpressionModel("/Users/sharad/Downloads//HackDukeML/venv/model.json", "/Users/sharad/Downloads//HackDukeML/venv/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture(0)
        self.video = cv2.VideoCapture("/Users/sharad/Downloads/zoom_call.mp4")
        #self.video = cv2.VideoCapture("/Users/sharad/Downloads/facial_expr.mp4")
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        #self. fps = self.video.get(cv2.CAP_PROP_FPS)
       # print("Frames per second", self.fps)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total_frames : ", self.total_frames)


    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        #if np.shape(fr) == ():
        #    return "empty"
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        #print("gray_fr", gray_fr)
        #print("len", len(gray_fr))
        #print("width", len(gray_fr[0]))
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        #print("faces", faces)
        #frames_per_sec = cv2.get(CAP_PROP_FPS)
        
        index = 0
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            #print("index", index)
            #print("pred", pred)
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            index += 1
        #print("count", count)
        #print("time", count/self.fps)


      
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

'''
vid = VideoCamera()

while True:
    vid.get_frame()

count = 0
#for i in range(vid.total_frames):
while True:
    rv = vid.get_frame()
    if rv == "empty":
        break
    count +=1

'''
  