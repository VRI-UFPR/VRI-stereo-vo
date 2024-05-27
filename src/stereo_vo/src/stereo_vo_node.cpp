#include "feature_matcher.hpp"

#include "rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "stereo_vo/srv/depth_estimator.hpp"
#include "stereo_vo/srv/feature_extractor.hpp"
#include "stereo_vo/srv/feature_matcher.hpp"

#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>

class StereoVONode : public rclcpp::Node
{
public:
    StereoVONode(void) : Node("stereo_vo")
    {
        RCLCPP_INFO(this->get_logger(), "Starting stereo visual odometry...");

        // Parse config
        std::string rcam_topic, lcam_topic;
        std::string rcam_intrinsics_file, lcam_intrinsics_file;
        std::string depth_estimator_service, feature_extractor_service, feature_matcher_service;
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["left_cam"]["topic"] >> lcam_topic;
        vo_config["right_cam"]["topic"] >> rcam_topic;
        vo_config["left_cam"]["intrinsics_file"] >> lcam_intrinsics_file;
        vo_config["right_cam"]["intrinsics_file"] >> rcam_intrinsics_file;
        vo_config["depth_estimator_service"] >> depth_estimator_service;
        vo_config["feature_extractor_service"] >> feature_extractor_service;
        vo_config["feature_matcher_service"] >> feature_matcher_service;
        fs.release();

        // Load camera intrinsics
        

        // Initialize subscribers and time synchronizer
        this->lcam_sub.subscribe(this, lcam_topic);
        this->rcam_sub.subscribe(this, rcam_topic);
        this->cam_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), lcam_sub, rcam_sub);
        this->cam_sync->setMaxIntervalDuration(rclcpp::Duration(0.01, 0));
        this->cam_sync->registerCallback(std::bind(&StereoVONode::cam_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Intialize clients
        this->depth_estimator_client = this->create_client<stereo_vo::srv::DepthEstimator>(depth_estimator_service);
        this->feature_extractor_client = this->create_client<stereo_vo::srv::FeatureExtractor>(feature_extractor_service);
        this->feature_matcher_client = this->create_client<stereo_vo::srv::FeatureMatcher>(feature_matcher_service);
    }

private:

    void waitForService(rclcpp::ClientBase::SharedPtr client)
    {
        while (!client->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
                rclcpp::shutdown();
            }
            RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
        }
    }

    void cam_callback(const sensor_msgs::msg::Image::SharedPtr lcam_msg, const sensor_msgs::msg::Image::SharedPtr rcam_msg)
    {
        auto depth_estimator_request = std::make_shared<stereo_vo::srv::DepthEstimator::Request>();
        depth_estimator_request->left_img = *lcam_msg;
        depth_estimator_request->right_img = *rcam_msg;
        waitForService(this->depth_estimator_client);

        auto depth_estimator_result = this->depth_estimator_client->async_send_request(depth_estimator_request);

        auto feature_extractor_request = std::make_shared<stereo_vo::srv::FeatureExtractor::Request>();
        feature_extractor_request->image = *lcam_msg;
        waitForService(this->feature_extractor_client);

        auto feature_extractor_result = this->feature_extractor_client->async_send_request(feature_extractor_request);
    }
    
    // Subscribers and time synchronizer
    message_filters::Subscriber<sensor_msgs::msg::Image> lcam_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> rcam_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cam_sync;

    // Clients
    rclcpp::Client<stereo_vo::srv::DepthEstimator>::SharedPtr depth_estimator_client;
    rclcpp::Client<stereo_vo::srv::FeatureExtractor>::SharedPtr feature_extractor_client;
    rclcpp::Client<stereo_vo::srv::FeatureMatcher>::SharedPtr feature_matcher_client;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVONode>());
  rclcpp::shutdown();
  return 0;
}
