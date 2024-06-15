cmake_minimum_required(VERSION 3.8)
project(stereo_vo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
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
find_package(vio_msgs REQUIRED)

# OpenCV dependencies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(cv_bridge REQUIRED)

# Add stereo_vo_node
add_executable(stereo_vo_node src/stereo_vo_node.cpp)
ament_target_dependencies(stereo_vo_node rclcpp message_filters sensor_msgs vio_msgs OpenCV cv_bridge)
target_link_libraries(stereo_vo_node ${OpenCV_LIBS})
install(TARGETS
  stereo_vo_node
  DESTINATION lib/${PROJECT_NAME})

# Add depth_estimator_server
add_executable(depth_estimator_server src/depth_estimator_server.cpp)
ament_target_dependencies(depth_estimator_server rclcpp sensor_msgs vio_msgs OpenCV cv_bridge)
target_link_libraries(depth_estimator_server ${OpenCV_LIBS})
install(TARGETS
  depth_estimator_server
  DESTINATION lib/${PROJECT_NAME})

# Add feature_extractor_server
add_executable(feature_extractor_server src/feature_extractor_server.cpp)
ament_target_dependencies(feature_extractor_server rclcpp sensor_msgs vio_msgs OpenCV cv_bridge)
target_link_libraries(feature_extractor_server ${OpenCV_LIBS})
install(TARGETS
  feature_extractor_server
  DESTINATION lib/${PROJECT_NAME})

# Copy launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
