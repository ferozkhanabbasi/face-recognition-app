
import numpy as np
import cv2
import pickle
import pyttsx3



FrontfaceClassifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eyeCascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
#face recognizers
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
i=0
#text to speech
engine = pyttsx3.init()

labels ={"person_name" : 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
   


def speak(text):
    engine.say("Hello Mister")
    engine.say(text)
    engine.runAndWait()




capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    gray = cv2.equalizeHist(frame)
    face = FrontfaceClassifier.detectMultiScale(gray, scaleFactor=1.5 , minNeighbors=5)
    
    for (x,y,w,h) in face:
        print(x,y,w,h) 
        region_of_intrst_gray = gray[y:y+h, x:x+w] # y,x cord = ycord height, xcord height= ystart, xstart        
        region_of_intrst_color = frame[y:y+h, x:x+w]
        id_ , conf = recognizer.predict(region_of_intrst_gray)
     
        if conf>=4 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            if i<1:
                speak(name)
                i=i+1

            color = (255,255,255)
            stroke = 1
            cv2.putText(frame, name , (x,y), font , 2, color, stroke, cv2.LINE_AA)
  
        save_img = "current_image.png" 
        cv2.imwrite( save_img ,  region_of_intrst_gray) 
        color = (25, 0, 0)  #blue
        weight = 2
        cord_x_end = x+w
        cord_y_end = y+h
        cv2.rectangle(frame , (x, y), (cord_x_end, cord_y_end), color, weight)
        eyes = eyeCascade.detectMultiScale(region_of_intrst_gray)
   #  smile = smileCascade.detectMultiScale(region_of_intrst_gray)
        
   #     for (ex, ey, ew, eh) in eyes:
    #        cv2.rectangle(region_of_intrst_color, (ex, ey) , (ex+ew , ey+eh), (0, 255,0), 2)
    #    for (sx, sy, sw, sh) in smile:
     #       cv2.rectangle(region_of_intrst_color, (sx, sy) , (sx+sw , sy+sh), (0, 255,0), 2)



    cv2.imshow('recognization',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()