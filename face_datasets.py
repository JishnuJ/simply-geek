#Project done by Jishnu J
#For more info visit jishnu.co
#this is a test
#this is another test
#this is another another test

import cv2
import os

def assure_path_exists(path): 
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

vid_cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = 1

count = 0

assure_path_exists("dataset/")

while(True):

    _, image_frame = vid_cam.read()

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count>100:
        break

vid_cam.release()

cv2.destroyAllWindows()
