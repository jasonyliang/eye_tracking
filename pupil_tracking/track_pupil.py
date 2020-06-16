import cv2
import dlib
import numpy as np
import time 

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
	points = [shape[i] for i in side]
	points = np.array(points, dtype=np.int32)
	mask = cv2.fillConvexPoly(mask, points, 255)
	return mask, points

def contouring(thresh, mid, img, right=False):
	cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	try:
		cnt = max(cnts, key = cv2.contourArea)
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		if right:
			cx += mid
		cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
		return cx, cy
	except:
		#pass
		return None, None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()
font = cv2.FONT_HERSHEY_PLAIN 

cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("calibration",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
	pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

def average_positions(pupil_positions):
	# get rid of the first 10% 
	start = int(len(pupil_positions) * 0.1)
	LX, LY, RX, RY = [],[],[],[]
	for (lx, ly, rx, ry) in pupil_positions[start:]:
		if lx:
			LX.append(lx)
		if ly:
			LY.append(ly)
		if rx:
			RX.append(rx)
		if ry:
			RY.append(ry)
	return np.mean(LX), np.mean(LY), np.mean(RX), np.mean(RY)


# set calibration
# need calibration function F to map positions of pupils p to positions on the screen s
# s_x, s_y = F(p_x, p_y)
# using SVR: we train two seaprate SVRs
p = []
s = [(10, 10), (1200, 10), (10, 600), (1200, 600)]
for screen_points in s:
	pupil_positions = []
	while True:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 1)
		cv2.circle(img, screen_points, 16, (0, 0, 255), 2)
		cv2.putText(img, f"Look at the Red Circles in the corner, move on to the next circle by pressing the key 'q'", 
			(500, 100), font, 1, (255,0,0))
		for rect in rects:

			shape = predictor(gray, rect)
			shape = shape_to_np(shape)
			mask = np.zeros(img.shape[:2], dtype=np.uint8)
			mask, left_points = eye_on_mask(mask, left)
			mask, right_points = eye_on_mask(mask, right)
			mask = cv2.dilate(mask, kernel, 5)
			eyes = cv2.bitwise_and(img, img, mask=mask)
			mask = (eyes == [0, 0, 0]).all(axis=2)
			eyes[mask] = [255, 255, 255]
			mid = (shape[42][0] + shape[39][0]) // 2
			eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
			threshold = 77 #cv2.getTrackbarPos('threshold', 'image')
			_, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
			thresh = cv2.erode(thresh, None, iterations=2) #1
			thresh = cv2.dilate(thresh, None, iterations=4) #2
			thresh = cv2.medianBlur(thresh, 3) #3
			thresh = cv2.bitwise_not(thresh)
			left_cx, left_cy = contouring(thresh[:, 0:mid], mid, img) # left eye
			right_cx, right_cy = contouring(thresh[:, mid:], mid, img, True) # right eye
			# print(f"Left eye at {left_points}, Right eye at {right_points}")
			# print(f"Left pupil at {left_cx, left_cy}, Right pupil at {right_cx, right_cy}")
			# for (x, y) in shape[36:48]:
			#     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
		# show the image with the face detections + facial landmarks
		#cv2.imshow('eyes', img)
		cv2.imshow('calibration', img)
		pupil_positions.append((left_cx, left_cy, right_cx, right_cy))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	print(f"Screen Position {screen_points}")

	print(f"Pupil Position {average_positions(pupil_positions)}")

# while(True):
# 	ret, img = cap.read()
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	rects = detector(gray, 1)
# 	for rect in rects:

# 		shape = predictor(gray, rect)
# 		shape = shape_to_np(shape)
# 		mask = np.zeros(img.shape[:2], dtype=np.uint8)
# 		mask = eye_on_mask(mask, left)
# 		mask = eye_on_mask(mask, right)
# 		mask = cv2.dilate(mask, kernel, 5)
# 		eyes = cv2.bitwise_and(img, img, mask=mask)
# 		mask = (eyes == [0, 0, 0]).all(axis=2)
# 		eyes[mask] = [255, 255, 255]
# 		mid = (shape[42][0] + shape[39][0]) // 2
# 		eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
# 		threshold = 77 #cv2.getTrackbarPos('threshold', 'image')
# 		_, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
# 		thresh = cv2.erode(thresh, None, iterations=2) #1
# 		thresh = cv2.dilate(thresh, None, iterations=4) #2
# 		thresh = cv2.medianBlur(thresh, 3) #3
# 		thresh = cv2.bitwise_not(thresh)
# 		left_cx, left_cy = contouring(thresh[:, 0:mid], mid, img) # left eye
# 		right_cx, right_cy = contouring(thresh[:, mid:], mid, img, True) # right eye
# 		print(f"Left eye at {left_cx, left_cy}, Right eye at {right_cx, right_cy}")
# 		# for (x, y) in shape[36:48]:
# 		#     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
# 	# show the image with the face detections + facial landmarks
# 	cv2.imshow('eyes', img)
# 	cv2.imshow("image", thresh)
# 	#time.sleep(10)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
	
cap.release()
cv2.destroyAllWindows()