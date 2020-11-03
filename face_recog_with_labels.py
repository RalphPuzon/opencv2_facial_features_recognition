#short script to demonstrate facial features detection.

#per line 36, pressing q exits out of program. change the 'q' for a different
#exit key.

import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_outline = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0 , 0), 2)
        cv2.putText(face_outline, 'Face', (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (252, 194, 1), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 7)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.2, 45)
        for (ex, ey, ew, eh) in eyes:
            eyes_outline = cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                                         (1, 194, 252), 2)
            cv2.putText(eyes_outline, 'Eye', (ex, ey-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (252, 194, 1), 2)
        for (sx, sy, sw, sh) in smile:
            mouth_outline= cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh),
                                         (1, 194, 252), 2)
            cv2.putText(mouth_outline, 'Mouth', (sx, sy-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (252, 194, 1), 2)
    return frame
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

        
