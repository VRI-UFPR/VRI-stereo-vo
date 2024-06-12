#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

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
    
    // Subscribers and time synchronizer
    message_filters::Subscriber<sensor_msgs::msg::Image> lcam_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> rcam_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cam_sync;

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
        RCLCPP_INFO(this->get_logger(), "Loading camera intrinsics from file: %s", intrinsics_file.c_str());

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

    sensor_msgs::msg::Image::SharedPtr undistortImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg, const CameraIntrinsics &intrinsics)
    {
        cv::Mat img = cv_bridge::toCvCopy(msg, "mono8")->image;
        cv::Mat undistorted_img;
        cv::undistort(img, undistorted_img, intrinsics.camera_matrix, intrinsics.dist_coeffs);

        return cv_bridge::CvImage(msg->header, "mono8", undistorted_img).toImageMsg();
    }

    void depth_callback(rclcpp::Client<vio_msgs::srv::DepthEstimator>::SharedFuture future) {
        auto status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::ready) 
        {
            RCLCPP_INFO(this->get_logger(), "Depth estimation completed.");

            service_done = true;
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Service In-Progress...");
        }
    }

    void feature_callback(rclcpp::Client<vio_msgs::srv::FeatureExtractor>::SharedFuture future) {
        auto status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::ready) 
        {
            RCLCPP_INFO(this->get_logger(), "Feature extraction completed.");

            service_done = true;
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Service In-Progress...");
        }
    }    

    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr &lcam_msg, const sensor_msgs::msg::Image::ConstSharedPtr &rcam_msg)
    {
        if (!service_done)
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received stereo image message. Size: %dx%d", lcam_msg->width, lcam_msg->height);

        vio_msgs::msg::StereoImage undistorted_stereo_img;
        // Undistort image and create stereo image message
        undistorted_stereo_img.left_image = *(this->undistortImage(lcam_msg, this->lcam_intrinsics));
        undistorted_stereo_img.right_image = *(this->undistortImage(rcam_msg, this->rcam_intrinsics));

        // Send depth estimation request
        // auto depth_estimator_request = std::make_shared<vio_msgs::srv::DepthEstimator::Request>();
        // depth_estimator_request->stereo_image = undistorted_stereo_img;
        // waitForService(this->depth_estimator_client);

        // this->service_done = false;
        // auto depth_estimator_result = this->depth_estimator_client->async_send_request(depth_estimator_request, std::bind(&StereoVONode::depth_callback, this, std::placeholders::_1));
        // RCLCPP_INFO(this->get_logger(), "Depth estimation request sent.");

        // Send feature extraction request
        auto feature_extractor_request = std::make_shared<vio_msgs::srv::FeatureExtractor::Request>();
        feature_extractor_request->image = undistorted_stereo_img.left_image;
        waitForService(this->feature_extractor_client);

        this->service_done = false;
        auto feature_extractor_result = this->feature_extractor_client->async_send_request(feature_extractor_request, std::bind(&StereoVONode::feature_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Feature extraction request sent.");
    }   

public:

    StereoVONode(void) : Node("stereo_vo")
    {
        RCLCPP_INFO(this->get_logger(), "Starting stereo visual odometry...");

        // Parse config
        std::string stereo_img_topic;
        std::string lcam_topic, rcam_topic;
        std::string rcam_intrinsics_file, lcam_intrinsics_file;
        std::string depth_estimator_service, feature_extractor_service, feature_matcher_service;
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["left_cam"]["topic"] >> lcam_topic;
        vo_config["right_cam"]["topic"] >> rcam_topic;
        vo_config["left_cam"]["intrinsics_file"] >> lcam_intrinsics_file;
        vo_config["right_cam"]["intrinsics_file"] >> rcam_intrinsics_file;
        vo_config["depth_estimator_params"]["depth_estimator_service"] >> depth_estimator_service;
        vo_config["feature_extractor_service"] >> feature_extractor_service;
        vo_config["feature_matcher_service"] >> feature_matcher_service;
        fs.release();

        // Load camera intrinsics
        this->loadCameraIntrinsics(lcam_intrinsics_file, this->lcam_intrinsics);
        this->loadCameraIntrinsics(rcam_intrinsics_file, this->rcam_intrinsics);

        // Initialize subscribers and time synchronizer
        this->lcam_sub.subscribe(this, lcam_topic);
        this->rcam_sub.subscribe(this, rcam_topic);        
        this->cam_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), lcam_sub, rcam_sub);
        this->cam_sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.04));
        this->cam_sync->registerCallback(std::bind(&StereoVONode::stereo_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Intialize clients
        this->depth_estimator_client = this->create_client<vio_msgs::srv::DepthEstimator>(depth_estimator_service);
        this->feature_extractor_client = this->create_client<vio_msgs::srv::FeatureExtractor>(feature_extractor_service);
        this->feature_matcher_client = this->create_client<vio_msgs::srv::FeatureMatcher>(feature_matcher_service);

        RCLCPP_INFO(this->get_logger(), "Stereo visual odometry node started.");
    }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVONode>());
  rclcpp::shutdown();
  return 0;
}
