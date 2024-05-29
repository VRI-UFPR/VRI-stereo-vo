#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/image.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "vio_msgs/srv/feature_extractor.hpp"
#include "vio_msgs/srv/feature_matcher.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>

class StereoVONode : public rclcpp::Node
{
private:

    bool service_done = true;
    
    // Subscriber
    rclcpp::Subscription<vio_msgs::msg::StereoImage>::SharedPtr stereo_img_sub;

    // Clients
    rclcpp::Client<vio_msgs::srv::DepthEstimator>::SharedPtr depth_estimator_client;
    rclcpp::Client<vio_msgs::srv::FeatureExtractor>::SharedPtr feature_extractor_client;
    rclcpp::Client<vio_msgs::srv::FeatureMatcher>::SharedPtr feature_matcher_client;

    // Camera intrinsics
    struct CameraIntrinsics
    {
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
    } lcam_intrinsics, rcam_intrinsics; 

    void loadCameraIntrinsics(const std::string &intrinsics_file, CameraIntrinsics &intrinsics)
    {
        cv::FileStorage fs(intrinsics_file, cv::FileStorage::READ);
        fs["K"] >> intrinsics.camera_matrix;
        fs["D"] >> intrinsics.dist_coeffs;
        fs.release();
    }

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

    sensor_msgs::msg::Image::SharedPtr undistortImage(const sensor_msgs::msg::Image msg, const CameraIntrinsics &intrinsics)
    {
        cv::Mat img = cv_bridge::toCvCopy(msg, "mono8")->image;
        cv::Mat undistorted_img;
        cv::undistort(img, undistorted_img, intrinsics.camera_matrix, intrinsics.dist_coeffs);

        // Save image for visualization
        cv::imwrite("/workspace/undistorted.png", undistorted_img);
        
        return cv_bridge::CvImage(msg.header, "mono8", undistorted_img).toImageMsg();
    }

    void depth_callback(rclcpp::Client<vio_msgs::srv::DepthEstimator>::SharedFuture future) {
        auto status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::ready) 
        {
            RCLCPP_INFO(this->get_logger(), "Depth estimation completed.");

            service_done = true;

            // Save depth image
            auto depth_map = future.get()->depth_map;
            cv::Mat depth_img = cv_bridge::toCvCopy(depth_map, "mono16")->image;
            cv::imwrite("/workspace/depth.png", depth_img);
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Service In-Progress...");
        }
    }

    void stereo_callback(const vio_msgs::msg::StereoImage::SharedPtr stereo_msg)
    {
        if (!service_done)
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received stereo image message. Size: %dx%d", stereo_msg->left_image.width, stereo_msg->left_image.height);

        vio_msgs::msg::StereoImage undistorted_stereo_img;
        // Undistort image and create stereo image message
        undistorted_stereo_img.left_image = *(this->undistortImage(stereo_msg->left_image, this->lcam_intrinsics));
        undistorted_stereo_img.right_image = *(this->undistortImage(stereo_msg->right_image, this->rcam_intrinsics));

        // Send depth estimation request
        auto depth_estimator_request = std::make_shared<vio_msgs::srv::DepthEstimator::Request>();
        depth_estimator_request->stereo_image = *stereo_msg;
        waitForService(this->depth_estimator_client);

        this->service_done = false;
        auto depth_estimator_result = this->depth_estimator_client->async_send_request(depth_estimator_request, std::bind(&StereoVONode::depth_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Depth estimation request sent.");
    }   

public:

    StereoVONode(void) : Node("stereo_vo")
    {
        RCLCPP_INFO(this->get_logger(), "Starting stereo visual odometry...");

        // Parse config
        std::string stereo_img_topic;
        std::string rcam_intrinsics_file, lcam_intrinsics_file;
        std::string depth_estimator_service, feature_extractor_service, feature_matcher_service;
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["stereo_image_topic"] >> stereo_img_topic;
        vo_config["left_cam_intrinsics"] >> lcam_intrinsics_file;
        vo_config["right_cam_intrinsics"] >> rcam_intrinsics_file;
        vo_config["depth_estimator_service"] >> depth_estimator_service;
        vo_config["feature_extractor_service"] >> feature_extractor_service;
        vo_config["feature_matcher_service"] >> feature_matcher_service;
        fs.release();

        // Load camera intrinsics
        this->loadCameraIntrinsics(lcam_intrinsics_file, this->lcam_intrinsics);
        this->loadCameraIntrinsics(rcam_intrinsics_file, this->rcam_intrinsics);

        // Initialize subscribers and time synchronizer
        this->stereo_img_sub = this->create_subscription<vio_msgs::msg::StereoImage>(stereo_img_topic, 10, std::bind(&StereoVONode::stereo_callback, this, std::placeholders::_1));

        // Intialize clients
        this->depth_estimator_client = this->create_client<vio_msgs::srv::DepthEstimator>(depth_estimator_service);
        this->feature_extractor_client = this->create_client<vio_msgs::srv::FeatureExtractor>(feature_extractor_service);
        this->feature_matcher_client = this->create_client<vio_msgs::srv::FeatureMatcher>(feature_matcher_service);
    }

    bool is_service_done() const 
    {
        return this->service_done;
    }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVONode>());
  rclcpp::shutdown();
  return 0;
}
