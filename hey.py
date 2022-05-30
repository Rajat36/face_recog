import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
from PIL import ImageGrab
from win32api import GetSystemMetrics

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

path = 'Images'
images = []
personNames = []
myList = os.listdir(path)

#print(myList)

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
#print(personNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')


while True:
    #ret, frame = cap.read()
    #faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    #faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    img = ImageGrab.grab(bbox=(0, 0, width, height))
    img_np = np.array(img)
    img_final = cv2.resize(img_np, (0, 0), None, 1,1)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)


    facesCurrentFrame = fr.face_locations(img_final)
    encodesCurrentFrame = fr.face_encodings(img_final, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            top, right, bottom, left = faceLoc
            cv2.rectangle(img_final, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img_final, (left, bottom +20 ), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img_final, name, (left , bottom +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),1)
            attendance(name)
    cv2.imshow('Screen Capture', img_final)


    if cv2.waitKey(1) == 13:
        break


cv2.destroyAllWindows()