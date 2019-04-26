import cv2
from random import randint 

class Draw:

	def __init__(self, output_name):
		self.output_name = output_name

	def setImage(self, image):
		self.image = image

	def addBox(self, xmin, ymin, xmax, ymax, id):
		x = xmin
		y = ymin 
		w = xmax - xmin
		h = ymax - ymin
		color = (randint(0, 255), randint(0, 255), randint(0, 255))
		cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
		text_size = cv2.getTextSize(id, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
		cv2.putText(self.image, id, (x, y + 10),
				      cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

	def displayImage(self):
		cv2.imshow(self.output_name, self.image)
		cv2.waitKey(1)

