/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "detector//network_config.h"

namespace config
{

// Common global vars
const bool kPRINT_PERF_INFO = false;
const bool kPRINT_PRED_INFO = false;
const bool kSAVE_DETECTIONS = true;

const std::string kPRECISION = "kFLOAT";
const std::string kINPUT_BLOB_NAME = "data";
const uint kINPUT_H = 416;
const uint kINPUT_W = 416;
const uint kINPUT_C = 3;
const uint64_t kINPUT_SIZE = kINPUT_C * kINPUT_H * kINPUT_W;
const uint kOUTPUT_CLASSES = 80;
const std::vector<std::string> kCLASS_NAMES
    = {"person",        "bicycle",       "car",           "motorbike",
       "aeroplane",     "bus",           "train",         "truck",
       "boat",          "traffic light", "fire hydrant",  "stop sign",
       "parking meter", "bench",         "bird",          "cat",
       "dog",           "horse",         "sheep",         "cow",
       "elephant",      "bear",          "zebra",         "giraffe",
       "backpack",      "umbrella",      "handbag",       "tie",
       "suitcase",      "frisbee",       "skis",          "snowboard",
       "sports ball",   "kite",          "baseball bat",  "baseball glove",
       "skateboard",    "surfboard",     "tennis racket", "bottle",
       "wine glass",    "cup",           "fork",          "knife",
       "spoon",         "bowl",          "banana",        "apple",
       "sandwich",      "orange",        "broccoli",      "carrot",
       "hot dog",       "pizza",         "donut",         "cake",
       "chair",         "sofa",          "pottedplant",   "bed",
       "diningtable",   "toilet",        "tvmonitor",     "laptop",
       "mouse",         "remote",        "keyboard",      "cell phone",
       "microwave",     "oven",          "toaster",       "sink",
       "refrigerator",  "book",          "clock",         "vase",
       "scissors",      "teddy bear",    "hair drier",    "toothbrush"};
const std::string kMODELS_PATH =   "/models/";

// Model V3 specific common global vars
const float kPROB_THRESH = 0.7f;
const float kNMS_THRESH = 0.5f;
const std::string kYOLO_CONFIG_PATH = "/data/yolov3.cfg";
const std::string kTRAINED_WEIGHTS_PATH = "/data/yolov3.weights";
const std::string kNETWORK_TYPE = "yolov3";
const uint kBBOXES = 3;
const std::vector<float> kANCHORS = {10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
                                     45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};


// Model V3 specific unique global vars
namespace yoloV3
{
const uint kSTRIDE_1 = 32;
const uint kSTRIDE_2 = 16;
const uint kSTRIDE_3 = 8;
const uint kGRID_SIZE_1 = kINPUT_H / kSTRIDE_1;
const uint kGRID_SIZE_2 = kINPUT_H / kSTRIDE_2;
const uint kGRID_SIZE_3 = kINPUT_H / kSTRIDE_3;
const uint64_t kOUTPUT_SIZE_1 = kGRID_SIZE_1 * kGRID_SIZE_1 * (kBBOXES * (5 + kOUTPUT_CLASSES));
const uint64_t kOUTPUT_SIZE_2 = kGRID_SIZE_2 * kGRID_SIZE_2 * (kBBOXES * (5 + kOUTPUT_CLASSES));
const uint64_t kOUTPUT_SIZE_3 = kGRID_SIZE_3 * kGRID_SIZE_3 * (kBBOXES * (5 + kOUTPUT_CLASSES));
const std::vector<int> kMASK_1 = {6, 7, 8};
const std::vector<int> kMASK_2 = {3, 4, 5};
const std::vector<int> kMASK_3 = {0, 1, 2};
const std::string kOUTPUT_BLOB_NAME_1 = "yolo_83";
const std::string kOUTPUT_BLOB_NAME_2 = "yolo_95";
const std::string kOUTPUT_BLOB_NAME_3 = "yolo_107";
} // namespace yoloV3


} // namespace config