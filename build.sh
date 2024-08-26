#!/bin/bash

cmake_args=$1

source /opt/ros/humble/install/setup.bash
colcon build --symlink-install --cmake-args "$cmake_args" --packages-select vio_msgs 
source install/setup.bash
colcon build --symlink-install --cmake-args "$cmake_args"