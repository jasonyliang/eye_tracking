import cv2
import numpy as np 
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_68.dat')

def mid_point(point1, point2):
	return int((point1.x + point2.x)/2), int ((point1.y+point2.y)/2)

while True:
	_, frame = cap.read()
	# grayscale the frames
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray)

	for face in faces:
		# x1, y1 = face.left(), face.top()
		# x2, y2 =  face.right(), face.bottom() 
		#cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
		landmarks = predictor(gray, face)
		# x, y = landmarks.part(36).x, landmarks.part(36).y
		# cv2.circle(frame, (x,y), 3, (0,0,255), 2)

		# detect the horizontal line for the left eye
		left_point = (landmarks.part(36).x, landmarks.part(36).y)
		right_point = (landmarks.part(39).x, landmarks.part(39).y)
		horizontal_line = cv2.line(frame, left_point, right_point, (0,255,0), 4)

		# detect the vertical line for the left eye
		top_point = mid_point(landmarks.part(37), landmarks.part(38))
		bottom_point = mid_point(landmarks.part(41), landmarks.part(40))
		vertical_line = cv2.line(frame, top_point, bottom_point, (0,255,0),4)
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1)

	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()