#!/usr/bin/env python
import rospy 
import sys
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageConverter: 

    def __init__(self):
        self.image_pub = rospy.Publisher("Jetson_Camera",Image, queue_size=1)
        self.bridge = CvBridge()
    
    
    def read_cam(self):
        width = 416
        height = 416
        # open Jetson Camera
        gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        # read image from camera
        if cap.isOpened():
            while True:
                _, img = cap.read()
                cv2.imshow("Jetson TX2 Camera", img)
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
                except CvBridgeError as e:
                    print(e)  
        else:
            print "Unable to open Cap"

                    
if __name__ == '__main__':
    ic = ImageConverter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        ic.read_cam()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
