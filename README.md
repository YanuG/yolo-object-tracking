# yolo_object_tracking

## Dependencies 
  * JetPack 3.3
  * OpenCV 
      
      Follow instructions in this [link](https://jkjung-avt.github.io/opencv3-on-tx2/) to install the correct version of OpenCV
  * ROS Kinetic 
      
      Follow instructions in this [link](https://www.jetsonhacks.com/2017/03/27/robot-operating-system-ros-nvidia-jetson-tx2/) to install the ROS on Jetson TX2 (Note: This works for Jetpack 3.3) 
  * Tensorflow 
      
       ```pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp33 tensorflow-gpu```

## Running the Project

Enable maximum clock speed. This is done in the home directory ``` ./jetson-clocks.sh```

Run roscore ```roscore```

Execute code ``` rosrun yolo_object_tracking yolo_object_tracking```

To change input image, in ```config/default_config.json``` change the parameter pathToImage.

## Current Results

Using TensorRT and Yolov3, Jetson has a framerate approximately 6 fps, this is double the current frame rate. 
