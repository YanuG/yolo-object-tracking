/* Yolo Detector headers */
#include "yolo_object_tracking/ds_image.h"
#include "yolo_object_tracking/network_config.h"
#include "yolo_object_tracking/trt_utils.h"
#include "yolo_object_tracking/yolo.h"
#include "yolo_object_tracking/yolov3.h"
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
#include <ros/package.h>
#include <boost/bind.hpp>
#include <sensor_msgs/Image.h>
#include "yolo_object_tracking/BoundingBoxesVector.h"
#include <cv_bridge/cv_bridge.h>


/**
 *  Describes the return variable from interface 
 **/
typedef struct {
    std::vector<BBoxInfo> remaining;
    double beginTimer;
    double endTimer;

} InterfaceObj;


/**
 * Converts the image type and sends the image to be anaylzed by the NN on the GPU. 
 * **/ 
InterfaceObj interface(DsImage &dsImage, std::unique_ptr<Yolo>& inferNet) {
    InterfaceObj interfaceObj; 
    cv::Mat trtInput = blobFromDsImages(dsImage, inferNet->getInputH(), inferNet->getInputW()); 
    interfaceObj.beginTimer = ros::Time::now().toSec();
    inferNet->doInference(trtInput.data);
    interfaceObj.endTimer  = ros::Time::now().toSec();
    auto binfo = inferNet->decodeDetections(0, dsImage.getImageHeight(), dsImage.getImageWidth());
    interfaceObj.remaining = nonMaximumSuppression(inferNet->getNMSThresh(), binfo);
    return interfaceObj; 
}

/**
 *  Used to test an image. This function will display predictions, save the image with prediciton
 *  boxes drawn on it and display time for inference. 
 **/
void test(nlohmann::json configFile, std::unique_ptr<Yolo>& inferNet) {
    cv::Mat image = cv::imread(configFile["test"]["pathToImage"], CV_LOAD_IMAGE_COLOR);
    DsImage dsImage = DsImage(image, inferNet->getInputH(), inferNet->getInputW()); 
    InterfaceObj interfaceObj =  interface(dsImage, inferNet);   
    for (auto b : interfaceObj.remaining) {
        if (configFile["test"]["displayPredicition"])
            printPredictions(b, inferNet->getClassName(b.label));
        dsImage.addBBox(b, inferNet->getClassName(b.label));
    }
    if (configFile["test"]["displayPredicition"])
        dsImage.showImage(5000);

    if (configFile["test"]["displayInferenceTime"])
        std::cout << "Inference time : " << interfaceObj.endTimer - interfaceObj.beginTimer  << " s" << std::endl;
}

/**
 * Callback function for Jetson Camera
**/
void cameraCallback(const sensor_msgs::ImageConstPtr& msg, nlohmann::json configFile, std::unique_ptr<Yolo>* inferNet, ros::Publisher pub)
{
    cv::Mat orginalImage = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::Mat image;
    cv::resize(orginalImage, orginalImage, cv::Size(540, 540)); 
    cv::flip(orginalImage, image, 0);
       if (configFile["camera"]["showRawImage"]){
         cv::imshow("image_raw", image);
         cv::waitKey(1);
     }
    DsImage dsImage = DsImage(image, (*inferNet) -> getInputH(), (*inferNet) -> getInputW()); 
    InterfaceObj interfaceObj =  interface(dsImage, (*inferNet)); 

    yolo_object_tracking::BoundingBoxes data;
    yolo_object_tracking::BoundingBoxesVector boxMsg;

    for (auto b : interfaceObj.remaining) {
        if (configFile["camera"]["displayPredicition"])
            printPredictions(b, (*inferNet)->getClassName(b.label));
        data.xmin = b.box.x1;  data.ymin = b.box.y1;  data.xmax = b.box.x2;  data.ymax = b.box.y2;  data.id = (*inferNet)->getClassName(b.label);
        boxMsg.boundingBoxesVector.push_back(data);
        dsImage.addBBox(b, (*inferNet)->getClassName(b.label));
    }

    if (configFile["camera"]["displayDetection"])
        dsImage.showImage(1);

    if (configFile["camera"]["displayInferenceTime"])
        std::cout << "Inference time : " << interfaceObj.endTimer - interfaceObj.beginTimer  << " s" << std::endl;  

    pub.publish(boxMsg);
}

int main(int argc, char** argv)
{
    // create a ROS handle 
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle nh;
    // sets a path to the configuration file 
    nlohmann::json configFile;
    std::string pathToJsonFile =  ros::package::getPath("yolo_object_tracking") + "/config/default_config.json";
    std::ifstream i(pathToJsonFile);
    i >> configFile;
    // inferface with YoloV3 NN
    std::unique_ptr<Yolo> inferNet = std::unique_ptr<Yolo>{new YoloV3(1)};
    // determine camera input
    std::string image_input = configFile["camera_input"];
    ros::Subscriber sub;
    ros::Publisher pub = nh.advertise<yolo_object_tracking::BoundingBoxesVector>("/boundingBoxes", 1);
    if (image_input == "test") {
        test(configFile, inferNet);
    } else if (image_input == "camera"){
        sub = nh.subscribe<sensor_msgs::Image>(configFile["camera"]["topicName"], 1, 
                boost::bind(&cameraCallback, _1, configFile, &inferNet, pub));
    } 
    ros::spin();
    return 0;
}