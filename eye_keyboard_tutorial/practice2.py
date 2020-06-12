# virtual keyboards
import cv2
import numpy as np 

keyboard = np.zeros((1000,1500,3), dtype=np.uint8)


def letter(x,y,text):
	# keys
	width = 200
	height = 200
	thick = 3
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

letter(0,0,"A")
letter(200,0,"B")
letter(0,200,"C")
letter(200,200,"D")

cv2.imshow("Keyboard", keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()

