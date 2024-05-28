cmake_minimum_required(VERSION 3.8)
project(drivers)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(vio_msgs REQUIRED)

# Include Cpp "include" directory
include_directories(include)

# Include CUDA
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# OpenCV dependencies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(jetson-utils REQUIRED)
set(JetsonUtils_LIBS "/usr/local/lib/libjetson-utils.so")

# camera_publisher
add_executable(camera_publisher src/camera_publisher.cpp src/image_converter.cpp src/ros_compat.cpp) 
ament_target_dependencies(camera_publisher rclcpp sensor_msgs vio_msgs vision_msgs jetson-utils OpenCV CUDA)
target_link_libraries( camera_publisher ${OpenCV_LIBS} ${JetsonUtils_LIBS})
install(TARGETS
  camera_publisher
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/bno_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()