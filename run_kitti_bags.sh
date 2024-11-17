#!/bin/bash

for i in 00 01 02 03 04 05 06 07 08 09 10
do
    timeout 500s ros2 launch stereo_vo vo.launch.py bagfile:=kitti_data_odometry_gray_sequence_$i-ros2_bag 2>&1 | tee -a Data/logs/kitti-log_$i.log
done