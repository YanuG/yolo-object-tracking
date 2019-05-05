# yolo_object_tracking

## Dependencies 
  * Jetson TX2
  * JetPack 3.3 
  * OpenCV 
      
      Follow instructions in this [link](https://jkjung-avt.github.io/opencv3-on-tx2/) to install the correct version of OpenCV
  * ROS Kinetic 
      
      Follow instructions in this [link](https://www.jetsonhacks.com/2017/03/27/robot-operating-system-ros-nvidia-jetson-tx2/) to install the ROS on Jetson TX2 

## Setting up workspace 

1. Create ros workspace. Follow instructions in this [tutoial](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). Call your workspace ```catkin_ws```

2. Navigate to ```catkin_ws/src```

3. Install ros packages [jetson_csi_cam](https://github.com/peter-moran/jetson_csi_cam.git) and [gscam](https://github.com/ros-drivers/gscam.git)

4. Clone the following repo

5. Install dependencies ```rosdep install --from-paths ../src -i -y``` 

6. Navigate back to ```catkin_ws``` and run ```catkin_make``` 

## Running the Project

1. Source workspace. ```source ~/catkin_ws/devel/setup.bash```

2. Enable maximum clock speed. This is done in the home directory ``` ./jetson-clocks.sh```

3. Run launch file ```roslaunch yolo_object_tracking cameraN.launch``` N represents the Jetson number within the coordinate system.  
