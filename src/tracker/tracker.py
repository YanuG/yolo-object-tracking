#!/usr/bin/env python
import multiprocessing
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import math
import shapely.geometry
#for ROS
import rospy 
from yolo_object_tracking.msg import BoundingBoxesVector
from cv_bridge import CvBridge
#for reading from config
import json
#for unique ids
import uuid

class Tracker():
	def __init__(self, settings):
		"""
		:param dict settings
		"""
		self.settings = settings
		self.currentInputDict = {}
		self.tracked_boxes = []
		(major, minor) = cv2.__version__.split(".")[:2]
		self.OPENCV_OBJECT_TRACKERS = {
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create,
			"goturn": cv2.TrackerGOTURN_create
		}
		# From : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = uuid.uuid4()
		self.objects = OrderedDict()
		self.objectMetaData = OrderedDict()
		self.disappeared = OrderedDict()
		# Camera id for camera handoff
		self.cameraID = self.settings.get('cameraID', 0)
		# size of the screen TODO read from json
		self.screenWidth = 416

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = self.settings.get('obj_disappear_thresh', 60)
		self.obj_teleport_threshold = self.settings.get('obj_teleport_threshold', 0.4)
		# add this to the init function 
		rospy.init_node('tracker', anonymous=True)
		# when a message is sent to this topic it will call the update method 
		rospy.Subscriber("/detector_values_0", BoundingBoxesVector, self.handleROSMessage)
		# create publisher 
		self.bridge = CvBridge()

	def register(self, centroid, rect):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		tracker = self.OPENCV_OBJECT_TRACKERS.get(self.settings['tracker'], cv2.TrackerKCF_create)()

		roiRect = self.rects_to_roi(rect)
		roituple = (roiRect[0],roiRect[1],roiRect[2],roiRect[3])
		# print(f"ROI Rect: {roituple}, type: {type(roituple)}")
		tracker.init(self.image, roituple)

		self.objectMetaData[self.nextObjectID] = [rect, tracker]
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID = uuid.uuid4()
		#self.nextObjectID += 1

	def handleROSMessage(self, msg):
		# handles the callback function from ROS to change the coordinate of the bounding boxes according to which screen it came from
		rects = {}
		self.image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")  
		for (i , boundingBoxes) in enumerate(msg.boundingBoxesVector): 
			if(self.cameraID == msg.feedID):
				rect = {'xmin': boundingBoxes.xmin + 416, 
                    'xmax': boundingBoxes.xmax + 416, 
                    'ymin': boundingBoxes.ymin, 
                    'ymax': boundingBoxes.ymax}
				rects[i] = rect;

	def deregister(self, objectIndex):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		print(objectIndex)
		del self.objects[objectIndex]
		del self.disappeared[objectIndex]
		del self.objectMetaData[objectIndex]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects.boundingBoxesVector) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			IDSToDeregister = []
			for objectID in self.disappeared.keys():
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					# self.deregister(objectID)
					IDSToDeregister.append(objectID)
			for objectID in IDSToDeregister:
				 self.deregister(objectID)
			# ROS publish disappeared object
			# return early as there are no centroids or tracking info
			# to update
			return 

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects.boundingBoxesVector), 2), dtype="int")

		# loop over the bounding box rectangles
		# print rects.boundingBoxesVector
		for (i , boundingBoxes) in enumerate(rects.boundingBoxesVector):
			# use the bounding box coordinates to derive the centroid
			cX = int((boundingBoxes.xmin + boundingBoxes.xmax) / 2.0)
			cY = int((boundingBoxes.ymin + boundingBoxes.ymax) / 2.0) 
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], rects.boundingBoxesVector[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			# print(self.objects)
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			print("objIDS: " , objectIDs)
			print("objCentroid: " , objectCentroids)

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			distance_mat = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the front of the index
			# list
			rows = distance_mat.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = distance_mat.argmin(axis=1)[rows]
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
				  continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				x1, y1 = self.objects[objectID][0], self.objects[objectID][1]
				x2, y2 = inputCentroids[col][0],inputCentroids[col][1]
				height, width, channels = self.image.shape
				# print(f'height: {height}, width: {width}, ch: {channels}')
				if math.hypot(x2-x1, y2-y1) < self.obj_teleport_threshold*(height+width)*0.5:
					self.objects[objectID] = inputCentroids[col]
					self.disappeared[objectID] = 0

					# indicate that we have examined each of the row and
					# column indexes, respectively
					usedRows.add(row)
					usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, distance_mat.shape[0])).difference(usedRows)
			unusedCols = set(range(0, distance_mat.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if distance_mat.shape[0] >= distance_mat.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					# if we choose to remove threshold we can remove this condition - Yanushka  
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col], rects.boundingBoxesVector[col])


	def rects_to_roi(self, rect):
		left, top, right, bottom = rect.xmin, rect.ymax, rect.xmax, rect.ymin 
		roibox = np.asarray([(left+right)/2, (top+bottom)/2, abs(right-left), abs(bottom-top)])
		return roibox.astype("int")

if __name__ == '__main__':
	# put settings in json file 
	with open('tracker_config.json', 'r') as trackerConfig:
		data = trackerConfig.read()
	settings = json.loads(data)
	trackerSettings = {'tracker': settings["tracker"],
					    'obj_disappear_thresh': settings["obj_disappear_thresh"],
					    'obj_teleport_threshold': settings["obj_teleport_threshold"], 
					    'cameraID': settings["cameraID"],
	}

	t = Tracker(
	settings=trackerSettings)
	rospy.spin()
