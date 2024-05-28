cmake_minimum_required(VERSION 3.8)
project(stereo_vo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Include Cpp "include" directory
include_directories(include)

ament_package()
