#!/usr/bin/env python
import sys
import multiprocessing
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import math
import shapely.geometry
#for ROS
import rospy 
from yolo_object_tracking.msg import BoundingBoxesVector, BoundingBoxes
from cv_bridge import CvBridge
#for reading from config
import json
#for unique ids
import uuid
#for displaying
import rospkg 

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
		self.objects = OrderedDict()
		self.objectMetaData = OrderedDict()
		self.disappeared = OrderedDict()
		# Camera id for camera handoff
		self.cameraID = self.settings.get('cameraID', 0)
		self.numCameras = self.settings.get('numCameras', 1)
		self.screenWidth = self.settings.get('width', 416)
		self.detectorTopic = self.settings.get('detectorTopic', "/detector_values_0")
		self.handoffSub = self.settings.get('handoffTopicSub', "/handoff0")
		self.handoffPub = self.settings.get('handoffTopicPub', "/handoff1")
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = self.settings.get('obj_disappear_thresh', 60)
		self.obj_teleport_threshold = self.settings.get('obj_teleport_threshold', 0.4)
		# add this to the init function 
		rospy.init_node('tracker', anonymous=True)
		# when a message is sent to this topic it will call the update method 
		rospy.Subscriber(self.detectorTopic, BoundingBoxesVector, self.update)
		rospy.Subscriber(self.handoffSub, BoundingBoxesVector, self.update)
		# create publisher 
		self.bridge = CvBridge()
		self.pub = rospy.Publisher(self.handoffPub, BoundingBoxesVector, queue_size=1)
		self.image = None

	def register(self, centroid, rect, objectUID):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[objectUID] = centroid
		tracker = self.OPENCV_OBJECT_TRACKERS.get(self.settings['tracker'], cv2.TrackerKCF_create)()

		roiRect = self.rects_to_roi(rect)
		roituple = (roiRect[0],roiRect[1],roiRect[2],roiRect[3])
		# print(f"ROI Rect: {roituple}, type: {type(roituple)}")
		# tracker.init(self.image, roituple)
		self.objectMetaData[objectUID] = [rect, tracker]
		self.disappeared[objectUID] = 0
		print "registered " + str(objectUID)

	def deregister(self, objectIndex):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		if(not(self.cameraID == 0 and self.objects[objectIndex][0] < 208) or (self.cameraID == self.numCameras-1 and self.objects[objectIndex][0] > 208)):
			#publish it
			box = BoundingBoxes()
			box.xmin, box.ymin, box.xmax, box.ymax, box.id = self.objects[objectIndex][0], self.objects[objectIndex][1], self.objects[objectIndex][0], self.objects[objectIndex][1], objectIndex
			bVector = BoundingBoxesVector()
			bVector.boundingBoxesVector.append(box)
			bVector.feedID = self.cameraID		
			self.pub.publish(bVector)
			print "published " + str(objectIndex)	

		del self.objects[objectIndex]
		del self.disappeared[objectIndex]
		del self.objectMetaData[objectIndex]

	def update(self, rects):
		# handles the callback function from ROS to change the coordinate of the bounding boxes according to which screen it came from
		incrementAmount = 0
		# update the coordinates depending on which feed it came in from
		if(self.cameraID > rects.feedID): # feed came in from the left camera
			incrementAmount = -(self.screenWidth*2)
		elif(self.cameraID < rects.feedID): # feed came in from the right camera
			incrementAmount = 2*self.screenWidthr*2
		else: # feed came in from its own detector
			incrementAmount = self.screenWidth*2	
		
		for (i , boundingBoxes) in enumerate(rects.boundingBoxesVector): 
			if self.cameraID == rects.feedID:
				objectUID = boundingBoxes.id + '-' + str(uuid.uuid4())[0:3]
			else:
				objectUID = boundingBoxes.id
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
		# print rects
		for (i , boundingBoxes) in enumerate(rects.boundingBoxesVector):
			# use the bounding box coordinates to derive the centroid
			cX = int((boundingBoxes.xmin + boundingBoxes.xmax ) / 2.0)
			cY = int((boundingBoxes.ymin + boundingBoxes.ymax) / 2.0) 
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], rects.boundingBoxesVector[i], objectUID)

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			# print(self.objects)
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			#print("objIDS: " , objectIDs)
			#print("objCentroid: " , objectCentroids)

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
				if self.image is not None:
					height, width, channels = 416, 416, 3
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
					# if we choose to remove threshold we can remove this condition
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
					self.register(inputCentroids[col], rects.boundingBoxesVector[col], objectUID)


	def rects_to_roi(self, rect):
		left, top, right, bottom = rect.xmin, rect.ymax, rect.xmax, rect.ymin 
		roibox = np.asarray([(left+right)/2, (top+bottom)/2, abs(right-left), abs(bottom-top)])
		return roibox.astype("int")

if __name__ == '__main__':
	# get project path
	rospack = rospkg.RosPack()
	path = rospack.get_path('yolo_object_tracking') + "/config/camera1_config.json"
	with open(path, 'r') as trackerConfig:
		data = trackerConfig.read()
	settings = json.loads(data)
	trackerSettings = {'tracker': settings["tracker"],
						'detectorTopic': settings["detectorTopic"],
						'obj_disappear_thresh': settings["obj_disappear_thresh"],
						'obj_teleport_threshold': settings["obj_teleport_threshold"], 
						'cameraID': settings["cameraID"],
						'width': settings["image"]["width"],
						'numCameras': settings["numCameras"],
						'detectorTopic':settings["detectorTopic"],
						'handoffTopicPub':settings["handoffTopicPub"],
						'handoffTopicSub':settings["handoffTopicSub"]

	}

	t = Tracker(
	settings=trackerSettings)
	rospy.spin()
   
