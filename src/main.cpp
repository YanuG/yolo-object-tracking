#include "yolo_object_tracking/ds_image.h"
#include "yolo_object_tracking/network_config.h"
#include "yolo_object_tracking/trt_utils.h"
#include "yolo_object_tracking/yolo.h"
#include "yolo_object_tracking/yolov3.h"

/* OpenCV headers */
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <experimental/filesystem>
#include <gflags/gflags.h>
#include <string>
#include <sys/time.h>

/* ROS headers */
#include "ros/ros.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle n;

    std::unique_ptr<Yolo> inferNet = std::unique_ptr<Yolo>{new YoloV3(1)};
    DsImage dsImage;
    cv::Mat image = cv::imread("/home/nvidia/Downloads/trump.jpeg", CV_LOAD_IMAGE_COLOR);
    dsImage = DsImage(image, inferNet->getInputH(), inferNet->getInputW()); 
    cv::Mat trtInput = blobFromDsImages(dsImage, inferNet->getInputH(), inferNet->getInputW()); 
    double m_begin = ros::Time::now().toSec();
    inferNet->doInference(trtInput.data);
    double m_end  = ros::Time::now().toSec();
    auto binfo = inferNet->decodeDetections(0, dsImage.getImageHeight(),
                                                        dsImage.getImageWidth());
    auto remaining = nonMaximumSuppression(inferNet->getNMSThresh(), binfo);
    for (auto b : remaining)
    {
        printPredictions(b, inferNet->getClassName(b.label));
        dsImage.addBBox(b, inferNet->getClassName(b.label));
    }
    dsImage.saveImageJPEG("/home/nvidia/catkin_ws/src/yolo_object_tracking/test/");
    std::cout << " Inference time per image : " << m_end - m_begin  << " s" << std::endl;

    return 0;
}