/* Yolo Detector headers */
#include "detector/ds_image.h"
#include "detector/network_config.h"
#include "detector/trt_utils.h"
#include "detector/yolo.h"
#include "detector/yolov3.h"
/* JSON header */
#include "utilities/json.hpp"
/* OpenCV headers */
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
/* ROS headers */
#include "ros/ros.h"
#include "yolo_object_tracking/BoundingBoxesVector.h"
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>

// header sequence number
int counter = 0;
// default values of an image size
int imageWidth = 0;
int imageHeight = 0;
/**
 * Callback function for Jetson csi cam
**/
void cameraCallback(const sensor_msgs::ImageConstPtr& msg, nlohmann::json configFile, std::unique_ptr<Yolo>* inferNet, ros::Publisher pub)
{   
    // increament sequence 
    counter ++;
    cv::Mat orginalImage = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::Mat image;
    cv::resize(orginalImage, orginalImage, cv::Size(imageWidth, imageHeight)); 
    // currently image is reotated from camera - need to fix that 
    cv::flip(orginalImage, image, 0);
       if (configFile["showRawImage"]){
         cv::imshow("image_raw", image);
         cv::waitKey(1);
    }
    // resize image
    DsImage dsImage = DsImage(image, (*inferNet) -> getInputH(), (*inferNet) -> getInputW()); 
    // covert image to a trtInput
    cv::Mat trtInput = blobFromDsImages(dsImage, (*inferNet)->getInputH(), (*inferNet)->getInputW()); 
    double beginTimer = ros::Time::now().toSec();
    // send image to the GPU
    (*inferNet)->doInference(trtInput.data);
    double endTimer  = ros::Time::now().toSec();
    //decode results
    auto binfo = (*inferNet)->decodeDetections(0, dsImage.getImageHeight(), dsImage.getImageWidth());
    std::vector<BBoxInfo> remaining = nonMaximumSuppression((*inferNet)->getNMSThresh(), binfo);
    // create ROS messages 
    yolo_object_tracking::BoundingBoxes data;
    yolo_object_tracking::BoundingBoxesVector boxMsg;
    // loop throught detections 
    for (auto b : remaining) {
        if (configFile["displayPredicition"])
            printPredictions(b, (*inferNet)->getClassName(b.label));
        // save bounding box and class name into ros message BoundingBoxes
        data.xmin = b.box.x1;  data.ymin = b.box.y1;  data.xmax = b.box.x2;  data.ymax = b.box.y2;  data.id = (*inferNet)->getClassName(b.label);
        boxMsg.boundingBoxesVector.push_back(data);
        if (configFile["drawOnImage"])
            // TODO - put in draw.py
            dsImage.addBBox(b, (*inferNet)->getClassName(b.label));
    }
    boxMsg.feedID = configFile["cameraID"];
    // convert image as a ROS message (sensor_msgs/Image)
    std_msgs::Header header; 
    header.seq = counter; 
    header.stamp = ros::Time::now();  
    // Display detection 
    if (configFile["displayDetection"])
        dsImage.showImage(1); 
    // Display time it take to tranfer image to GPU, anaylsis it and return image 
    if (configFile["displayInferenceTime"])
        std::cout << "Inference time : " << endTimer - beginTimer  << " s" << std::endl; 
    // publish boundBoxesVector msg
    pub.publish(boxMsg);
}

int main(int argc, char** argv)
{
    // create a ROS handle 
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle nh;
    // Get current dir
    std::string pathToDir = ros::package::getPath("yolo_object_tracking");
    // Sets a path to the configuration file 
    std::string pathToJsonFile =  pathToDir + "/config/camera1_config.json";
    std::ifstream i(pathToJsonFile);
    nlohmann::json configFile;
    i >> configFile;
    // inferface with YoloV3 NN
    std::unique_ptr<Yolo> inferNet = std::unique_ptr<Yolo>{new YoloV3(1 , pathToDir)};
    // set image size 
    imageWidth = configFile["image"]["width"];
    imageHeight = configFile["image"]["height"];
    // create publishers and subscriers
    ros::Publisher pub = nh.advertise<yolo_object_tracking::BoundingBoxesVector>(configFile["detectorTopic"], 1);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>(configFile["cameraTopic"], 1, 
            boost::bind(&cameraCallback, _1, configFile, &inferNet, pub));
     
    ros::spin();
    return 0;
}
