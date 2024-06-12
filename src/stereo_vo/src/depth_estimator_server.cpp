#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "depth_estimator.hpp"

class DepthEstimatorServer : public rclcpp::Node
{
private:

    rclcpp::Service<vio_msgs::srv::DepthEstimator>::SharedPtr depth_server;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub;

    std::shared_ptr<DepthEstimator> depth_estimator;

    // Camera intrinsics
    struct CameraIntrinsics
    {
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
    } lcam_intrinsics, rcam_intrinsics; 

    void depth_callback(const std::shared_ptr<vio_msgs::srv::DepthEstimator::Request> request,
                        std::shared_ptr<vio_msgs::srv::DepthEstimator::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received depth estimation request. Size: " 
            << request->stereo_image.left_image.width << "x" << request->stereo_image.left_image.height << ", " 
            << request->stereo_image.right_image.width << "x" << request->stereo_image.right_image.height);

        cv::Mat img_left = cv_bridge::toCvCopy(request->stereo_image.left_image, "mono8")->image;
        cv::Mat img_right = cv_bridge::toCvCopy(request->stereo_image.right_image, "mono8")->image;

        auto estimation_start = std::chrono::high_resolution_clock::now();
        
        cv::Mat depth_map = this->depth_estimator->compute(img_left, img_right);

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Depth estimation time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());

        // Publish depth image for visualization
        cv::Mat depth_map_viz;
        cv::normalize(depth_map, depth_map_viz, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        this->depth_image_pub->publish(*(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", depth_map_viz).toImageMsg()));

        response->depth_map = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", depth_map).toImageMsg());

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

        // Initialize publishers
        this->depth_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/stereo_vo/depth_image", 10);

        // Initialize stereo BM
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
