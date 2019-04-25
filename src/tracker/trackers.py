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
#for reading from config
import configparser


class Tracker():
    def __init__(self, idx, inputq, outputq, stop_event, settings):
        """
        :param list idx: integer list to distinguish objects [type,sub-type,index]
        :param multiprocessing.Manager.Queue() inputq: each element is a dictionary with keys,
        'original_image','frame_id',['boxes'],['scores']['classes']
        boxes: list of [[y_min,x_min,y_max,x_max], [y_min,..],...] where
        | x_min, y_min |   |             |
        |              |   |             |
        |              |   | x_max,y_max |
        other possibly used names for coordinates within this class:
        | left, top    |   |             |
        |              |   |             |
        |              |   | right,bottom|
        scores: list of probabilities for each detection
        'classes': list of classes for each detection
        :param multiprocessing.Manager.Queue() outputq: Each element is a dictionary with keys,
        'idx'(=self.idx), 'frame_id'(=inputq.get()['frame_id']), ['boxes'],['scores'],['classes'] for tracked items
        :param multiprocessing.Event() stop_event: Signals the process to stop
        :param dict settings: {'tracker' : <tracker_type>, 'class_names':<class_names>}
        <tracker_type> : Possible values, 'csrt', 'kcf','boosting','mil','tld','medianflow','mosse'
        <class_names> : List of class names
        """
        multiprocessing.Process.__init__(self, name='tracker_'+str(idx))
        self.name = 'tracker_'+str(idx)
        self.idx = idx
        self.inputQ = inputq
        self.outputQ = outputq
        self.stopEvent = stop_event
        self.settings = settings
        self.currentInputDict = {}
        self.tracked_boxes = []
        (major, minor) = cv2.__version__.split(".")[:2]
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create,
            "goturn": cv2.TrackerGOTURN_create
        }
        # self.tracker = self.OPENCV_OBJECT_TRACKERS.get(self.settings['tracker'], cv2.TrackerKCF_create)()
        # self.multi_trackers = cv2.MultiTracker_create()
        self.class_names = self.settings['class_names']
        # self.ctracker = CentroidTracker(maxDisappeared=30)

        # From : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objectMetaData = OrderedDict()
        self.disappeared = OrderedDict()
        self.track_within_polygon = self.settings.get('track_within_polygon', False)
        self.tracked_polygon = shapely.geometry.polygon.Polygon(self.settings['tracked_polygon'])

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = self.settings.get('obj_disappear_thresh', 60)
        self.obj_teleport_threshold = self.settings.get('obj_teleport_threshold', 0.4)
        # add this to the init function 
        rospy.init_node('tracker', anonymous=True)
        # when a message is sent to this topic it will call the update method 
        rospy.Subscriber("/boundingBoxes", BoundingBoxesVector, update)

    def run(self):
        print('STARTING Tracker')
        while not self.stopEvent.is_set():
            # print('Tracker waiting for input')
            while not self.inputQ.empty():
                # print(f'Tracker got input {self.inputQ.qsize()}')
                if self.inputQ.qsize() > 10:
                    print('Input queue size is: {self.inputQ.qsize()}')
                    # Do something about this if this queue size is too large.
                    # This could be because this class can't process data in the same speed
                    # that it gets them. Maybe use only the latest set of data and discard
                    # the rest.
                    # EG:
                    print('Clearing queue...')
                    while self.inputQ.qsize() > 1: #or 1 if inputQ.get() is called again
                        self.currentInputDict = self.inputQ.get()
                self.currentInputDict = self.inputQ.get()
                self.tracked_boxes = []
                # mulithtreading
                for i, c in reversed(list(enumerate(self.currentInputDict['classes']))):
                    if c != self.idx[1]:  # idx[1] is the class type to track
                        continue
                    box = self.currentInputDict['boxes'][i]
                    top, left, bottom, right = box
                    ctbox = np.asarray([left, top, right, bottom])
                    self.tracked_boxes.append(ctbox.astype("int"))
                    #add ROS message (for loop to get ROS, add/minus 50 for box)

                #for i in reversed(list(enumerate(self.currentInputDict['classes']))):
                #if its from left
                    box = self.currentInputDict['boxes'][i]
                    top, left, bottom, right = box
                    ctbox = np.asarray    
                objects, total_tracked = self.update(self.tracked_boxes)
                classes = self.currentInputDict['classes']
                frameid = self.currentInputDict['frame_id']
                # print(f'Output from : {self.idx} , {frameid} = {classes}')
                # for class_member in objects:
                #     print(f'class_member: {class_member}')
                outputDict = {'idx': self.idx, 'frame_id': frameid,
                              'boxes': [], 'scores': [], 'classes': [],
                              'objects': objects, 'total_items': total_tracked}  # ##########TEMP##########
                self.outputQ.put(outputDict)
        # Do cleanup (if any)

    def register(self, centroid, rect):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        tracker = self.OPENCV_OBJECT_TRACKERS.get(self.settings['tracker'], cv2.TrackerKCF_create)()
        # print(f"Image type: {type(self.currentInputDict['original_image'])}")

        roiRect = self.rects_to_roi(rect)
        # (r1,r2,r3,r4) = roiRect
        roituple = (roiRect[0],roiRect[1],roiRect[2],roiRect[3])
        # print(f"ROI Rect: {roituple}, type: {type(roituple)}")
        tracker.init(self.currentInputDict['original_image'], roituple)
        # self.multi_trackers.add(tracker, self.currentInputDict['original_image'], self.rects_to_roi(rect))
        self.objectMetaData[self.nextObjectID] = [rect, tracker]
        # print(f"metadata updated. Metadata: {self.objectMetaData}")
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.objectMetaData[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if self.track_within_polygon:
            rects.boundingBoxesVector = [rect.boundingBoxesVector for rect.boundingBoxesVector in rects.boundingBoxesVector if
                     self.tracked_polygon.contains(shapely.geometry.Point(((rect.boundingBoxesVector[0]/2+rect.boundingBoxesVector[2]/2),
                                                                           (rect.boundingBoxesVector[1]/2+rect.boundingBoxesVector[3]/2))))]
        # self.updateCVTrackers()
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
                # send ROS message 
            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.nextObjectID

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects.boundingBoxesVector), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects.boundingBoxesVector):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        # print (self.objects)
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
            print("objIDS: ", objectIDs)
	    print("objCentroids: ", objectCentroids)

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
                height, width, channels = self.currentInputDict['original_image'].shape
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


        # return the set of trackable objects
        return self.objects, self.nextObjectID

    def rects_to_roi(self, rect):
        left, top, right, bottom = rect
        roibox = np.asarray([(left+right)/2, (top+bottom)/2, abs(right-left), abs(bottom-top)])
        return roibox.astype("int")

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    trackerOutQs = {}
    trackerInQs = {}  
    trackerStopEvent = manager.Event()
    trackerInQs[0] = {}
    trackerOutQs[0] = manager.Queue()
    classNames = ['person\n', 'bicycle\n', 'car\n', 'motorbike\n', 'aeroplane\n', 'bus\n', 'train\n', 'truck\n', 'boat\n', 'traffic light\n', 'fire hydrant\n', 'stop sign\n', 'parking meter\n', 'bench\n']
    trackerSettings = {'tracker': 'kcf',
                            'class_names': classNames,
                            'obj_disappear_thresh': 10,
                            'obj_teleport_threshold': 0.3,
                            'track_within_polygon': False,
                            'tracked_polygon': [(185, 0), (253, 224), (223, 476),
                                                (442, 479), (411, 224), (719, 206), (719, 0)]}

    t = Tracker(
        idx=[1,0,0], 
        inputq=trackerInQs[feedID][yoloclass],
        outputq=trackerOutQs[feedID], 
        stop_event=trackerStopEvent,
        settings=trackerSettings)
    t.start()



