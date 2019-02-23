/* Yolo Detector headers */
#include "yolo_object_tracking/ds_image.h"
#include "yolo_object_tracking/network_config.h"
#include "yolo_object_tracking/trt_utils.h"
#include "yolo_object_tracking/yolo.h"
#include "yolo_object_tracking/yolov3.h"
/* JSON header */
#include "yolo_object_tracking/json.hpp"
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
void test(nlohmann::json congfigFile, std::unique_ptr<Yolo>& inferNet) {
    cv::Mat image = cv::imread(congfigFile["test"]["pathToImage"], CV_LOAD_IMAGE_COLOR);
    DsImage dsImage = DsImage(image, inferNet->getInputH(), inferNet->getInputW()); 
    InterfaceObj interfaceObj =  interface(dsImage, inferNet);   
    for (auto b : interfaceObj.remaining) {
        if (congfigFile["test"]["displayPredicition"])
            printPredictions(b, inferNet->getClassName(b.label));
        dsImage.addBBox(b, inferNet->getClassName(b.label));
    }
    if (congfigFile["test"]["saveImageJPEG"]) {
        std::string pathToOutputImage =  ros::package::getPath("yolo_object_tracking") + "/output";
        dsImage.saveImageJPEG(pathToOutputImage);
    } 
    if (congfigFile["test"]["displayInferenceTime"])
        std::cout << "Inference time : " << interfaceObj.endTimer - interfaceObj.beginTimer  << " s" << std::endl;
}

/**
 * Callback function for Jetson Camera
**/
void cameraCallback(const sensor_msgs::ImageConstPtr& msg, nlohmann::json congfigFile, std::unique_ptr<Yolo>* inferNet)
{
    //std::unique_ptr<Yolo> inferNet = data;
    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    DsImage dsImage = DsImage(image, (*inferNet) -> getInputH(), (*inferNet) -> getInputW()); 
    InterfaceObj interfaceObj =  interface(dsImage, (*inferNet));   
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
    std::string image_imput = configFile["camera_input"];
    if (image_imput == "test"){
        test(configFile, inferNet);
    } else if (image_imput == "camera"){
        ros::Subscriber sub = n.subscribe<sensor_msgs::Image> ("Jetson_Camera/image", 1, 
                                   boost::bind(&cameraCallback, _1, configFile, &inferNet));
    } 
    return 0;
}