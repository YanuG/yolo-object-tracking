import cv2
from random import randint 

class Draw:

    def __init__(self, image, output_name):
        self.image = image
        self.output_name = output_name

    def addBox(self, xmin, ymin, xmax, ymax, id):
        x = xmin
        y = ymin 
        w = xmax - ymax
        h = ymax - ymin
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color)
        text_size = cv2.getTextSize(id, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
        print
        cv2.rectangle(self.image, (x + w, y + h) , (text_size.width + 3, text_size.height + 4), color, -1)
        cv2.putText(self.image, id, (x, y + text_size.height),
                     cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1, cv2.CV_AA)
    
    def displayImage(self):
        cv2.imshow(self.output_name, self.image)
        cv2.waitKey(1)



 