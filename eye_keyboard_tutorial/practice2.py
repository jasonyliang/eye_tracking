# virtual keyboards
import cv2
import numpy as np 

keyboard = np.zeros((600,1000,3), dtype=np.uint8)

keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}

def letter(indx,text, light):
	# Keys
	if indx == 0:
		x = 0
		y = 0
	elif indx == 1:
		x = 200
		y = 0
	elif indx == 2:
		x = 400
		y = 0
	elif indx == 3:
		x = 600
		y = 0
	elif indx == 4:
		x = 800
		y = 0
	elif indx == 5:
		x = 0
		y = 200
	elif indx == 6:
		x = 200
		y = 200
	elif indx == 7:
		x = 400
		y = 200
	elif indx == 8:
		x = 600
		y = 200
	elif indx == 9:
		x = 800
		y = 200
	elif indx == 10:
		x = 0
		y = 400
	elif indx == 11:
		x = 200
		y = 400
	elif indx == 12:
		x = 400
		y = 400
	elif indx == 13:
		x = 600
		y = 400
	elif indx == 14:
		x = 800
		y = 400
	# keys
	width = 200
	height = 200
	thick = 3
	if light:
		cv2.rectangle(keyboard, (x+thick,y+thick), (x+width-thick, y+height-thick ), (255,255,255), -1)
	else:
		cv2.rectangle(keyboard, (x+thick,y+thick), (x+width-thick, y+height-thick ), (255,0,0), 3)

	# text setting
	font = cv2.FONT_HERSHEY_PLAIN
	# text = letter
	font_scale = 10
	font_thick = 4
	text_size = cv2.getTextSize(text, font, font_scale, font_thick)
	text_width, text_height = text_size[0]

	text_x = int((width - text_width) / 2) + x
	text_y = int((height + text_height) / 2) + y

	cv2.putText(keyboard, text, (text_x,text_y), font, font_scale, (255,0,0), font_thick)

# another letter
# cv2.rectangle(keyboard, (200+thick,y+thick), (200+width-thick, y+height-thick ), (255,0,0), 3)
# cv2.putText(keyboard, "B", (text_x+200,text_y), font, font_scale, (255,0,0), font_thick)

for i in range(15):
    if i == 5:
        light = True
    else:
        light = False
    letter(i, keys_set_1[i], light)

cv2.imshow("Keyboard", keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()

