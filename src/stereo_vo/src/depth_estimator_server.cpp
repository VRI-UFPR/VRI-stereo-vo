#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "opencv_conversions.hpp"

#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "depth_estimator.hpp"

class DepthEstimatorServer : public rclcpp::Node
{
private:

    rclcpp::Service<vio_msgs::srv::DepthEstimator>::SharedPtr depth_server;

    std::shared_ptr<DepthEstimator> depth_estimator;

    void depth_callback(const std::shared_ptr<vio_msgs::srv::DepthEstimator::Request> request,
                        std::shared_ptr<vio_msgs::srv::DepthEstimator::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received depth estimation request. Size: " 
            << request->left_image.width << "x" << request->left_image.height << ", " 
            << request->right_image.width << "x" << request->right_image.height);
        auto estimation_start = std::chrono::high_resolution_clock::now();

        cv::Mat img_left = OpenCVConversions::toCvImage(request->left_image, "mono8");
        cv::Mat img_right = OpenCVConversions::toCvImage(request->right_image, "mono8");
        
        cv::Mat depth_map = this->depth_estimator->compute(img_left, img_right);

        // Normalize
        cv::normalize(depth_map, depth_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        response->depth_map = OpenCVConversions::toRosImage(depth_map, "mono8");

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_DEBUG(this->get_logger(), "Depth estimation time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());
        RCLCPP_INFO(this->get_logger(), "Depth estimation completed.");
    }

public:
    DepthEstimatorServer(): Node("depth_estimator_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting depth estimator server...");

        // Parse parameters
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode de_config = fs["stereo_vo"]["depth_estimator_params"];
        std::string depth_estimator_service = de_config["depth_estimator_service"].string();

        // Initialize service
        this->depth_server = this->create_service<vio_msgs::srv::DepthEstimator>(depth_estimator_service, 
            std::bind(&DepthEstimatorServer::depth_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize depth estimator
        std::string de_algorithm = de_config["depth_algorithm"];
        this->depth_estimator = std::make_shared<DepthEstimator>(de_config);
        
        fs.release();

        RCLCPP_INFO(this->get_logger(), "Depth estimator server started.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthEstimatorServer>());
    rclcpp::shutdown();
    return 0;
}
