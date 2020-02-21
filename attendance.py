import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font


window = tk.Tk()
window.geometry('600x500')
window.title("Attandance Menu")
window.configure(background='silver')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message = tk.Label(window, text="Attandance Management System Using Face Recognition" ,bg="silver"  ,fg="Brown"  ,width=50  ,height=3,font=('times', 13,  'bold underline')) 
message.place(x=30, y=20)

lbl1 = tk.Label(window, text="New ID",width=10  ,height=0  ,fg="black"  ,bg="gray" ,font=('times', 13, ' bold ') ) 
lbl1.place(x=25, y=100)

txt = tk.Entry(window, width=20, bg="white" ,fg="black",font=('times', 13, ' bold '))
txt.place(x=170, y=100)

lbl2 = tk.Label(window, text="Name", width=10 ,fg="black"  ,bg="gray"    ,height=0 ,font=('times', 13, ' bold ')) 
lbl2.place(x=25, y=135)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('times', 13, ' bold ')  )
txt2.place(x=170, y=135)


lbl4 = tk.Label(window, text="Section",width=10  ,height=0  ,fg="black"  ,bg="gray" ,font=('times', 13, ' bold ') ) 
lbl4.place(x=25, y=170)

txt4 = tk.Entry(window, width=20  ,bg="white" ,fg="black",font=('times', 13, ' bold '))
txt4.place(x=170, y=170)


lbl5 = tk.Label(window, text="Date",width=10  ,height=0  ,fg="black"  ,bg="gray" ,font=('times', 13, ' bold ') ) 
lbl5.place(x=25, y=205)

txt5 = tk.Entry(window, width=20  ,bg="white" ,fg="black",font=('times', 13, ' bold '))
txt5.place(x=170, y=205)

lbl3 = tk.Label(window, text="Status : ",width=10  ,fg="black"  ,bg="gray"  ,font=('times', 13, 'bold')) 
lbl3.place(x=25, y=240)
message = tk.Label(window, text="" ,bg="white"  ,fg="green"  ,width=30  , activebackground = "red" ,font=('times', 13, ' bold ')) 
message.place(x=170, y=240)



def fun():
    res = "Hi, Feroz"

def close_window ():
    window.destroy()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    Id = str(Id)
    Id = int(Id)
    name=(txt2.get())
    section=(txt4.get())
    date1=(txt5.get())
    faces,Id2 = getImagesAndLabels("TrainingImage")
    if Id in Id2:
        print("yes")
    else:
        print("no")
    if(is_number(Id) and name.isalpha() and Id not in Id2):
        cam = cv2.VideoCapture(0)
        harcascadePath = "cascades\haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)         
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+name +"."+str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>50:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + str(Id) +" Name : "+ name
        row = [Id , name , section , date1]
        with open('StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if Id in Id2:
            res = "You can not repeat an id"
            message.configure(text= res)
        if(is_number(Id)):
            res = "Enter a numeric id"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter an alphabetical name"
            message.configure(text= res)
        if(section.isalpha()):
            res = "Choose a section"
            message.configure(text= res)
        
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "cascades\haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("faceTrainner.yml")
    res = "Successfully Registered"
    message.configure(text= res)



def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    #print(imagePaths)
    
    faces=[]
    #create empty ID, Faces list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids



def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("faceTrainner.yml")
    harcascadePath = "cascades\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX    
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    i=0
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]    
                i=i+1            
            else:
                Id='Unknown'
                tt=str(Id)  
            if(conf > 75):
                cv2.imwrite("UnknownImage"+".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)   
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('image',im)
        key1= cv2.waitKey(10)   #escape button to close camera
        if(key1==27):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    print(attendance)
    res=attendance
    message.configure(text= res)   
takeImg = tk.Button(window, text="Take Image", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 13, ' bold '))
takeImg.place(x=30, y=330)
trainImg = tk.Button(window, text="Register", command=TrainImages  ,fg="black"  ,bg="green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 13, ' bold '))
trainImg.place(x=255, y=330)
trackImg = tk.Button(window, text="Mark Attendance", command=TrackImages  ,fg="black"  ,bg="brown"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 13, ' bold '))
trackImg.place(x=30, y=410)

 
window.mainloop()   