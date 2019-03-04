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
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

/**
 *  Describes the return variable from interface 
 **/
typedef struct {
    std::vector<BBoxInfo> remaining;
    double beginTimer;
    double endTimer;

} InterfaceInfo;

/**
 * Converts the image type and sends the image to be anaylzed by the NN on the GPU. 
 * **/ 
InterfaceInfo interface(DsImage &dsImage, std::unique_ptr<Yolo>& inferNet) {
    InterfaceInfo interfaceInfo; 
    cv::Mat trtInput = blobFromDsImages(dsImage, inferNet->getInputH(), inferNet->getInputW()); 
    interfaceInfo.beginTimer = ros::Time::now().toSec();
    inferNet->doInference(trtInput.data);
    interfaceInfo.endTimer  = ros::Time::now().toSec();
    auto binfo = inferNet->decodeDetections(0, dsImage.getImageHeight(), dsImage.getImageWidth());
    interfaceInfo.remaining = nonMaximumSuppression(inferNet->getNMSThresh(), binfo);
    return interfaceInfo; 
}

/**
 *  Used to test an image. This function will display predictions, save the image with prediciton
 *  boxes drawn on it and display time for inference. 
 **/
void test(nlohmann::json congfigFile, std::unique_ptr<Yolo>& inferNet) {
    cv::Mat image = cv::imread(congfigFile["test"]["pathToImage"], CV_LOAD_IMAGE_COLOR);
    DsImage dsImage = DsImage(image, inferNet->getInputH(), inferNet->getInputW()); 
    InterfaceInfo interfaceInfo =  interface(dsImage, inferNet);   
    for (auto b : interfaceInfo.remaining) {
        if (congfigFile["test"]["displayPredicition"])
            printPredictions(b, inferNet->getClassName(b.label));
        dsImage.addBBox(b, inferNet->getClassName(b.label));
    }
    if (congfigFile["test"]["displayImage"]) {
       dsImage.showImage();
    } 
    if (congfigFile["test"]["displayInferenceTime"])
        std::cout << "Inference time : " << interfaceInfo.endTimer - interfaceInfo.beginTimer  << " s" << std::endl;
}

/**
 * Callback function for Jetson Camera
**/
void cameraCallback(const sensor_msgs::ImageConstPtr& msg, nlohmann::json congfigFile, std::unique_ptr<Yolo>* inferNet)
{
    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::resize(image, image, cv::Size(384, 216));
    cv::flip(image, image, 0);
    cv::imshow("Image_Raw", image);
    cv::waitKey(1);
    DsImage dsImage = DsImage(image, (*inferNet) -> getInputH(), (*inferNet) -> getInputW()); 
    InterfaceInfo interfaceInfo =  interface(dsImage, (*inferNet));   

    if (congfigFile["camera"]["displayInferenceTime"])
        std::cout << "Inference time : " << interfaceInfo.endTimer - interfaceInfo.beginTimer  << " s" << std::endl; 
    for (auto b : interfaceInfo.remaining) {
        if (congfigFile["camera"]["displayPredicition"])
            printPredictions(b, (*inferNet) ->getClassName(b.label));
        dsImage.addBBox(b, (*inferNet)->getClassName(b.label));
    }
    if (congfigFile["test"]["displayImage"]) {
       dsImage.showImage();
    }
}


int main(int argc, char** argv)
{
    // create a ROS handle 
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle n;
    // sets a path to the configuration file 
    nlohmann::json configFile;
    std::string pathToJsonFile =  ros::package::getPath("yolo_object_tracking") + "/config/default_config.json";
    std::cout << pathToJsonFile << std::endl;
    std::ifstream i(pathToJsonFile);
    i >> configFile;
    // inferface with YoloV3 NN
    std::unique_ptr<Yolo> inferNet = std::unique_ptr<Yolo>{new YoloV3(1)};
    // determine camera input
    std::string image_input = configFile["camera_input"];
    ros::Subscriber sub;
    if (!image_input.compare("test")){
        test(configFile, inferNet);
    } else if (!image_input.compare("camera")){      
	sub = n.subscribe<sensor_msgs::Image> ("/csi_cam_0/image_raw", 1, 
                                   boost::bind(&cameraCallback, _1, configFile, &inferNet));
    }  
    ros::spin();
    return 0;
}
