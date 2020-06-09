import cv2
import numpy as np 
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()


while True:
	_, frame = cap.read()
	# grayscale the frames
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray)

	for face in faces:
		x1, y1 = face.left(), face.top()
		x2, y2 =  face.right(), face.bottom() 
		cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1)

	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()