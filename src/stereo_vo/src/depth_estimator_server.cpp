#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

class DepthEstimatorServer : public rclcpp::Node
{
public:
    DepthEstimatorServer(): Node("depth_estimator_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting depth estimator server...");

        // Parse parameters
        std::string depth_estimator_service; 
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["depth_estimator_service"] >> depth_estimator_service;
        fs.release();

        // Initialize service
        this->depth_server = this->create_service<stereo_vo::srv::DepthEstimator>(depth_estimator_service, 
            std::bind(&DepthEstimatorServer::depth_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Depth estimator server started.");
    }

private:

    void depth_callback(const std::shared_ptr<stereo_vo::srv::DepthEstimator::Request> request,
                        std::shared_ptr<stereo_vo::srv::DepthEstimator::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received depth estimation request. Size: " 
            << request->stereo_image.left_image.width << "x" << request->stereo_image.left_image.height << ", " 
            << request->stereo_image.right_image.width << "x" << request->stereo_image.right_image.height);

        cv::Size img_size(request->stereo_image.left_image.width, request->stereo_image.left_image.height);
        cv::Mat depth_map = cv::Mat::zeros(img_size, CV_32F);

        // TODO: Implement depth estimation

        respose->depth_map = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", depth_map).toImageMsg();
    }
    
    rclcpp::Service<stereo_vo::srv::DepthEstimator>::SharedPtr depth_server;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthEstimatorServer>());
    rclcpp::shutdown();
    return 0;
}
