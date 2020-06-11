import cv2
import numpy as np 
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_68.dat')
font = cv2.FONT_HERSHEY_PLAIN 	

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

def mid_point(point1, point2):
	return int((point1.x + point2.x)/2), int ((point1.y+point2.y)/2)

def get_blinking_ratio(face_landmarks, points):
	# detect the horizontal line for the left eye
		left_point = (face_landmarks.part(points[0]).x, face_landmarks.part(points[0]).y)
		right_point = (face_landmarks.part(points[3]).x, face_landmarks.part(points[3]).y)
		#horizontal_line = cv2.line(frame, left_point, right_point, (0,255,0), 4)

		# detect the vertical line for the left eye
		top_point = mid_point(face_landmarks.part(points[1]), face_landmarks.part(points[2]))
		bottom_point = mid_point(face_landmarks.part(points[5]), face_landmarks.part(points[4]))
		#vertical_line = cv2.line(frame, top_point, bottom_point, (0,255,0),4)

		vertical_line_length = hypot((top_point[0]-bottom_point[0]), (top_point[1]-bottom_point[1]))
		horizontal_line_length = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1]))

		ratio = horizontal_line_length/vertical_line_length # ratio above 4.5 seems  to  mean i'm blinking
		return ratio 
while True:
	_, frame = cap.read()
	# grayscale the frames
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
	# print(frame.shape)

	for face in faces:
		# x1, y1 = face.left(), face.top()
		# x2, y2 =  face.right(), face.bottom() 
		#cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
		landmarks = predictor(gray, face)
		# x, y = landmarks.part(36).x, landmarks.part(36).y
		# cv2.circle(frame, (x,y), 3, (0,0,255), 2)

		left_ratio = get_blinking_ratio(landmarks, left)
		right_ratio = get_blinking_ratio(landmarks, right)
		average_ratio = (left_ratio + right_ratio) /  2
		#if left_ratio > 5 or right_ratio > 5:
		if average_ratio > 5:
			# blinked
			cv2.putText(frame, "BLINKING", (50, 150), font, 1, (255,0,0))
		# Gaze detection
		left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
		                            (landmarks.part(37).x, landmarks.part(37).y),
		                            (landmarks.part(38).x, landmarks.part(38).y),
		                            (landmarks.part(39).x, landmarks.part(39).y),
		                            (landmarks.part(40).x, landmarks.part(40).y),
		                            (landmarks.part(41).x, landmarks.part(41).y)], 
		                            dtype=np.int32)
		#cv2.polylines(frame, np.int32([left_eye_region]), True, (0,0,255), 2)
		# make mask
		height, width, _ = frame.shape
		mask = np.zeros((height, width), np.uint8)
		cv2.polylines(mask, np.int32([left_eye_region]), True, 255, 2)
		cv2.fillPoly(mask, [left_eye_region], 255)
		left_eye = cv2.bitwise_and(gray, gray, mask=mask)



		min_x = np.min(left_eye_region[:, 0])
		max_x = np.max(left_eye_region[:, 0])
		min_y = np.min(left_eye_region[:, 1])
		max_y = np.max(left_eye_region[:, 1])

		eye = frame[min_y:max_y, min_x:max_x]
		gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
		_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)


		eye = cv2.resize(eye, None, fx=5, fy=5)
		threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)



		cv2.imshow("Eye", eye)
		cv2.imshow("Threshold", threshold_eye)
		#cv2.imshow("Mask", mask)
		cv2.imshow("Left Eye", left_eye)

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1)

	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()