#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "sensor_msgs/msg/image.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "vio_msgs/srv/feature_extractor.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>

class StereoVONode : public rclcpp::Node
{
private:

    bool depth_complete = true;
    bool feat_complete = true;
    cv::Mat curr_img = cv::Mat(), prev_img = cv::Mat();
    cv::Mat prev_img_desc = cv::Mat();
    std::vector<cv::KeyPoint> prev_keypoints;
    
    // Subscribers and time synchronizer
    message_filters::Subscriber<sensor_msgs::msg::Image> lcam_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> rcam_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cam_sync;

    // Visualization publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_map_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_map_pub;
    bool enable_viz = false;

    // Clients
    rclcpp::Client<vio_msgs::srv::DepthEstimator>::SharedPtr depth_estimator_client;
    rclcpp::Client<vio_msgs::srv::FeatureExtractor>::SharedPtr feature_extractor_client;

    // Camera intrinsics
    struct CameraIntrinsics
    {
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
    } lcam_intrinsics, rcam_intrinsics; 

    cv::Mat fromImgMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        return cv_bridge::toCvCopy(msg, "mono8")->image;
    }

    sensor_msgs::msg::Image::SharedPtr toImgMsg(const cv::Mat &img)
    {
        return cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", img).toImageMsg();
    }

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

    cv::Mat drawFeaturesSideBySide(const std::vector<int> &curr_img_points, const std::vector<int> &prev_img_points)
    {
        cv::Mat feat_img;
        if (this->prev_img.empty() || this->curr_img.empty())
        {
            return feat_img;
        }

        cv::hconcat(this->prev_img, this->curr_img, feat_img);
        cv::cvtColor(feat_img, feat_img, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < curr_img_points.size(); i += 2)
        {
            cv::Point2f curr_pt(curr_img_points[i], curr_img_points[i + 1]);
            cv::Point2f prev_pt(prev_img_points[i], prev_img_points[i + 1]);
            cv::line(feat_img, curr_pt, cv::Point2f(prev_pt.x + prev_img.cols, prev_pt.y), cv::Scalar(0, 255, 0), 1);
        }

        return feat_img;
    }

    void depth_callback(rclcpp::Client<vio_msgs::srv::DepthEstimator>::SharedFuture future) {
        auto status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::ready) 
        {
            RCLCPP_INFO(this->get_logger(), "Depth estimation completed.");

            this->depth_complete = true;

            auto response = future.get();
            if (this->enable_viz)
            {
                this->depth_map_pub->publish(response->depth_map);
            }
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Depth estimation in-progress...");
        }
    }

    void feature_callback(rclcpp::Client<vio_msgs::srv::FeatureExtractor>::SharedFuture future) {
        auto status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::ready) 
        {
            RCLCPP_INFO(this->get_logger(), "Feature extraction completed.");

            this->feat_complete = true;

            auto response = future.get();
            
            // Convert data from response
            cv::Mat curr_img_desc = cv_bridge::toCvCopy(response->curr_img_desc, "mono8")->image;
            std::vector<cv::KeyPoint> curr_keypoints;
            std::vector<cv::DMatch> good_matches;
            
            std::vector<int> curr_img_points = response->curr_img_points.data;
            std::vector<int> prev_img_points = response->prev_img_points.data;
            std::vector<int> distances = response->distances.data;
            std::vector<int> curr_keypoints_data = response->curr_img_keypoints.data;

            // Convert to opencv
            for (size_t i = 0; i < curr_img_points.size(); i += 2)
            {
                good_matches.push_back(cv::DMatch(curr_img_points[i], prev_img_points[i], distances[i]));
            }
            for (size_t i = 0; i < prev_img_points.size(); i += 2)
            {
                curr_keypoints.push_back(cv::KeyPoint(curr_keypoints_data[i], curr_keypoints_data[i + 1], 1));
            }

            if (this->enable_viz && !(this->prev_img.empty() || this->curr_img.empty()))
            {
                cv::Mat feat_img;
                cv::drawMatches(this->prev_img, this->prev_keypoints, this->curr_img, curr_keypoints, good_matches, feat_img, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                this->feature_map_pub->publish(*(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", feat_img).toImageMsg()));
            }
            
            this->prev_img_desc = curr_img_desc;
            this->prev_keypoints = curr_keypoints;
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Feature mathcer in-progress...");
        }
    }    

    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr &lcam_msg, const sensor_msgs::msg::Image::ConstSharedPtr &rcam_msg)
    {
        if (!this->depth_complete || !this->feat_complete)
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received stereo image message. Size: %dx%d", lcam_msg->width, lcam_msg->height);

        // Undistort image and create stereo image message
        sensor_msgs::msg::Image rect_limg = *(this->undistortImage(lcam_msg, this->lcam_intrinsics));
        sensor_msgs::msg::Image rect_rimg = *(this->undistortImage(rcam_msg, this->rcam_intrinsics));

        // Send depth estimation request
        auto depth_estimator_request = std::make_shared<vio_msgs::srv::DepthEstimator::Request>();
        depth_estimator_request->left_image = rect_limg;
        depth_estimator_request->right_image = rect_rimg;
        waitForService(this->depth_estimator_client);
        this->depth_estimator_client->async_send_request(depth_estimator_request, std::bind(&StereoVONode::depth_callback, this, std::placeholders::_1));
        this->depth_complete = false;
        RCLCPP_INFO(this->get_logger(), "Depth estimation request sent.");

        // Send feature matching request
        auto feature_extractor_request = std::make_shared<vio_msgs::srv::FeatureExtractor::Request>();
        feature_extractor_request->curr_img = rect_limg;
        feature_extractor_request->prev_img_desc = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", this->prev_img_desc).toImageMsg());
        waitForService(this->feature_extractor_client);
        this->feature_extractor_client->async_send_request(feature_extractor_request, std::bind(&StereoVONode::feature_callback, this, std::placeholders::_1));
        this->feat_complete = false;
        this->prev_img = curr_img;
        this->curr_img = cv_bridge::toCvCopy(rect_limg, "mono8")->image;
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
        std::string depth_estimator_service, feature_extractor_service;
        std::string depth_map_viz_topic, feature_viz_topic;
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["left_cam"]["topic"] >> lcam_topic;
        vo_config["right_cam"]["topic"] >> rcam_topic;
        vo_config["left_cam"]["intrinsics_file"] >> lcam_intrinsics_file;
        vo_config["right_cam"]["intrinsics_file"] >> rcam_intrinsics_file;
        vo_config["depth_estimator_params"]["depth_estimator_service"] >> depth_estimator_service;
        vo_config["feature_extractor_service"] >> feature_extractor_service;
        vo_config["debug_visualization"] >> this->enable_viz;
        vo_config["depth_map_viz"] >> depth_map_viz_topic;
        vo_config["feature_viz"] >> feature_viz_topic;
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

        if (this->enable_viz)
        {
            this->depth_map_pub = this->create_publisher<sensor_msgs::msg::Image>(depth_map_viz_topic, 10);
            this->feature_map_pub = this->create_publisher<sensor_msgs::msg::Image>(feature_viz_topic, 10);
        }

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
