#!/usr/bin/env python
import rospy 
import sys
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_convertor: 

    def __init__(self):
        self.image_pub = rospy.Publisher("Jetson_Cam",Image)
        self.bridge = CvBridge()

    def read_cam(self):
        cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1 ! nvtee ! nvvidconv flip-method=2 ! video/x-raw(memory:NVMM), format=(string)I420 ! nvoverlaysink -e ! appsink")
        if cap.isOpened():
            while True:
                _, img = cap.read()
                cv2.imshow("Image window", img)
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
                except CvBridgeError as e:
                    print(e)
    

if __name__ == '__main__':
    ic = image_convertor()
    rospy.init_node('image_converter', anonymous=True)
    try:
        ic.read_cam()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
