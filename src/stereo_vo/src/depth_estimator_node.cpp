#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "opencv_conversions.hpp"

#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "depth_estimator.hpp"

class DepthEstimatorNode : public rclcpp::Node
{
private:

    std::shared_ptr<DepthEstimator> depth_estimator;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher;

     // Subscribers and time synchronizer
    message_filters::Subscriber<sensor_msgs::msg::Image> lcam_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> rcam_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cam_sync;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscriber;

    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr &lcam_msg, const sensor_msgs::msg::Image::ConstSharedPtr &rcam_msg)
    {
        auto estimation_start = std::chrono::high_resolution_clock::now();

        // Convert ROS images to OpenCV images
        cv::Mat img_left = OpenCVConversions::toCvImage(*lcam_msg, "mono8");
        cv::Mat img_right = OpenCVConversions::toCvImage(*rcam_msg, "mono8");
        
        // Compute depth map
        cv::Mat depth_map = this->depth_estimator->compute(img_left, img_right);
        
        // Comvert to ROS message and publish
        sensor_msgs::msg::Image depth_msg = OpenCVConversions::toRosImage(depth_map, "mono8");
        depth_msg.header.stamp = lcam_msg->header.stamp; // Use left camera timestamp to syncronize later
        this->depth_publisher->publish(depth_msg);

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO_THROTTLE(this->get_logger(), *(this->get_clock()), 5000, "Depth estimation time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());
    }

public:
    DepthEstimatorNode(): Node("depth_estimator_node")
    {
        RCLCPP_INFO(this->get_logger(), "Starting depth estimator node...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config_imx.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);
        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        
        // Parse parameters
        cv::FileNode de_config = fs["stereo_vo"]["depth_estimator_params"];
        std::string topic = de_config["topic"].string();
        std::string lcam_topic = fs["stereo_vo"]["left_cam"]["topic"].string();
        std::string rcam_topic = fs["stereo_vo"]["right_cam"]["topic"].string();
        std::string lcam_intrinsics_file = fs["stereo_vo"]["left_cam"]["intrinsics_file"].string();
        std::string rcam_intrinsics_file = fs["stereo_vo"]["right_cam"]["intrinsics_file"].string();

        // Initialize publisher
        this->depth_publisher = this->create_publisher<sensor_msgs::msg::Image>(topic, 10);

        // Initialize subscribers and time synchronizer
        this->lcam_sub.subscribe(this, lcam_topic);
        this->rcam_sub.subscribe(this, rcam_topic);        
        this->cam_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), lcam_sub, rcam_sub);
        this->cam_sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.03));
        this->cam_sync->registerCallback(std::bind(&DepthEstimatorNode::stereo_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize depth estimator
        std::string de_algorithm = de_config["depth_algorithm"];
        this->depth_estimator = std::make_shared<DepthEstimator>(de_config, lcam_intrinsics_file, rcam_intrinsics_file);
        
        fs.release();

        RCLCPP_INFO(this->get_logger(), "Depth estimator node started.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthEstimatorNode>());
    rclcpp::shutdown();
    return 0;
}
