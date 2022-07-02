import cv2      #//OpenCv\\\

faceXML = cv2.CascadeClassifier('face.xml')    #choos your Option(face,body)

cap = cv2.VideoCapture(1)  # choos your camera(0,1)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceXML.detectMultiScale(gray)

    for (x, y, w, h) in faces:              
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('Detection Face...', frame)      #Showing Window

    k = cv2.waitKey(1) & 0xFF    #Stop
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()