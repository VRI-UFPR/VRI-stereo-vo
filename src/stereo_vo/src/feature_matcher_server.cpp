#include "rclcpp/rclcpp.hpp"

#include "stereo_vo/srv/feature_matcher.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <string>

#include <opencv2/opencv.hpp>

class FeatureMatcherServer : public rclcpp::Node
{
public:
    FeatureMatcherServer(): Node("feature_matcher_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting feature matcher server...");

        // Parse parameters
        std::string feature_matcher_service; 
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["feature_matcher_service"] >> feature_matcher_service;
        fs.release();

        // Initialize service
        this->feature_server = this->create_service<stereo_vo::srv::FeatureMatcher>(feature_matcher_service, 
            std::bind(&FeatureMatcherServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Feature matcher server started.");
    }

private:

    void feature_callback(const std::shared_ptr<stereo_vo::srv::FeatureMatcher::Request> request,
                        std::shared_ptr<stereo_vo::srv::FeatureMatcher::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature matching request. Size: " 
            << request->current_image.width << "x" << request->current_image.height << ", " 
            << request->previous_image.width << "x" << request->previous_image.height);
            
        response->matches = vision_msgs::Detection2DArray();
    }
    
    rclcpp::Service<stereo_vo::srv::FeatureMatcher>::SharedPtr feature_server;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FeatureMatcherServer>());
    rclcpp::shutdown();
    return 0;
}