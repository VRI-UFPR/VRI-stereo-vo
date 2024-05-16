#include "feature_matcher.hpp"

#include "rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <string>

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
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["left_cam"]["topic"] >> lcam_topic;
        vo_config["right_cam"]["topic"] >> rcam_topic;
        vo_config["left_cam"]["intrinsics_file"] >> lcam_intrinsics_file;
        vo_config["right_cam"]["intrinsics_file"] >> rcam_intrinsics_file;
        fs.release();

        // Load camera intrinsics
        // TODO: Load camera intrinsics from file

        // Create feature matcher
        // TODO: Create feature matcher

        // Create time synchronizer
        this->lcam_sub.subscribe(this, lcam_topic);
        this->rcam_sub.subscribe(this, rcam_topic);
        this->cam_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), lcam_sub, rcam_sub);
        this->cam_sync->setMaxIntervalDuration(rclcpp::Duration(0.01, 0));
        this->cam_sync->registerCallback(std::bind(&StereoVONode::cam_callback, this, std::placeholders::_1, std::placeholders::_2));
    }

private:

    void cam_callback(const sensor_msgs::msg::Image::SharedPtr lcam_msg, const sensor_msgs::msg::Image::SharedPtr rcam_msg)
    {
        
    }
    
    message_filters::Subscriber<sensor_msgs::msg::Image> lcam_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> rcam_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cam_sync;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVONode>());
  rclcpp::shutdown();
  return 0;
}
