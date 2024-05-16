#include "rclcpp/rclcpp.hpp"

#include "stereo_vo/srv/feature_extractor.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/detection_2d_array.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

class FeatureExtractorServer : public rclcpp::Node
{
public:
    FeatureExtractorServer(): Node("feature_extractor_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting feature extractor server...");

        // Parse parameters
        std::string feature_extractor_service; 
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["feature_extractor_service"] >> feature_extractor_service;
        fs.release();

        // Initialize service
        this->feature_server = this->create_service<stereo_vo::srv::FeatureExtractor>(feature_extractor_service, 
            std::bind(&FeatureExtractorServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Feature extractor server started.");
    }

private:

    void feature_callback(const std::shared_ptr<stereo_vo::srv::FeatureExtractor::Request> request,
                        std::shared_ptr<stereo_vo::srv::FeatureExtractor::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature extraction request. Size: " 
            << request->image.width << "x" << request->image.height);
            
        response->matches = vision_msgs::Detection2DArray();
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FeatureExtractorServer>());
    rclcpp::shutdown();
    return 0;
}