cmake_minimum_required(VERSION 3.8)
project(stereo_vo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(USE_CUDA  "Set to ON if the project is being built for running in a desktop device." ON)
if (USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()

# Include Cpp "include" directory
include_directories(include)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Messages
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vio_msgs REQUIRED)

# OpenCV dependencies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(cv_bridge REQUIRED)
find_package(yaml-cpp REQUIRED)

# Add stereo_vo_node
add_executable(stereo_vo_node src/stereo_vo_node.cpp)
ament_target_dependencies(stereo_vo_node rclcpp message_filters sensor_msgs vio_msgs nav_msgs OpenCV cv_bridge yaml-cpp)
target_link_libraries(stereo_vo_node ${OpenCV_LIBS} yaml-cpp::yaml-cpp)
install(TARGETS
  stereo_vo_node
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/visualization_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Copy launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
