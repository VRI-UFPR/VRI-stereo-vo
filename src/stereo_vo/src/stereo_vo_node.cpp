#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/Odometry.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "vio_msgs/srv/feature_extractor.hpp"
#include "vio_msgs/msg/d_matches.hpp"
#include "vio_msgs/msg/key_points.hpp"

#include "opencv_conversions.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>

class StereoVONode : public rclcpp::Node
{
private:

    Eigen::Matrix4d curr_pose = Eigen::Matrix4d::Identity();

    // Stereo vo state variables
    cv::Mat curr_img = cv::Mat(), prev_img = cv::Mat();
    cv::Mat prev_img_desc = cv::Mat();
    cv::Mat curr_depth_map = cv::Mat();
    std::vector<cv::KeyPoint> prev_img_kps, curr_img_kps;
    std::vector<cv::DMatch> good_matches;
    
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
    bool depth_complete = true;
    bool feat_complete = true;

    rclcpp::Publisher<geometry_msgs::msg::Odometry>::SharedPtr odometry_pub;

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

    cv::Mat undistortImage(const cv::Mat &img, const CameraIntrinsics &intrinsics)
    {
        cv::Mat undistorted_img;
        cv::undistort(img, undistorted_img, intrinsics.camera_matrix, intrinsics.dist_coeffs);

        return undistorted_img;
    }

    void publishOdometry(const Eigen::Matrix4d &pose)
    {
        geometry_msgs::msg::Odometry odometry_msg;
        odometry_msg.header.stamp = this->now();

        odometry_msg.pose.pose.position.x = pose(0, 3);
        odometry_msg.pose.pose.position.y = pose(1, 3);
        odometry_msg.pose.pose.position.z = pose(2, 3);

        Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
        odometry_msg.pose.pose.orientation.x = q.x();
        odometry_msg.pose.pose.orientation.y = q.y();
        odometry_msg.pose.pose.orientation.z = q.z();
        odometry_msg.pose.pose.orientation.w = q.w();

        this->odometry_pub->publish(odometry_msg);
    }

    void motionEstimation()
    {
        double cx = this->lcam_intrinsics.camera_matrix.at<double>(0, 2);
        double cy = this->lcam_intrinsics.camera_matrix.at<double>(1, 2);
        double fx = this->lcam_intrinsics.camera_matrix.at<double>(0, 0);
        double fy = this->lcam_intrinsics.camera_matrix.at<double>(1, 1);

        // Calculate previous image 3D points
        std::vector<cv::Point3f> pts_3d;
        for (auto p : this->prev_img_kps)
        {
            double z = this->curr_depth_map.at<float>(p.pt.y, p.pt.x);
            double x = (p.pt.x - cx) * z / fx;
            double y = (p.pt.y - cy) * z / fy;

            pts_3d.push_back(cv::Point3f(x, y, z));
        }

        // Solve PnP
        cv::Mat rvec, tvec;        
        cv::solvePnPRansac(pts_3d, this->curr_img_kps, this->lcam_intrinsics.camera_matrix, this->lcam_intrinsics.dist_coeffs, rvec, tvec);

        // Convert to transformation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R.ptr<double>());
        T.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(tvec.ptr<double>());

        this->curr_pose = this->curr_pose.dot(T);

        RCLCPP_INFO(this->get_logger(), "Estimated pose: \n%s", this->curr_pose.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")).c_str());

        this->publishOdometry(this->curr_pose);
    }

    void depth_callback(rclcpp::Client<vio_msgs::srv::DepthEstimator>::SharedFuture future) {
        auto status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::ready) 
        {
            RCLCPP_INFO(this->get_logger(), "Depth estimation completed.");

            auto response = future.get();
            this->curr_depth_map = OpenCVConversions::toCvImage(response->depth_map);

            if (this->enable_viz)
            {
                this->depth_map_pub->publish(response->depth_map);
            }

            if (this->feat_complete)
            {
                this->motionEstimation();
            }

            this->depth_complete = true;
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

            auto response = future.get();
            
            cv::Mat curr_img_desc = OpenCVConversions::toCvImage(response->curr_img_desc);
            this->curr_img_kps = OpenCVConversions::toCvKeyPoints(response->curr_keypoints);
            this->good_matches = OpenCVConversions::toCvDMatches(response->good_matches);

            if (this->enable_viz && !(this->prev_img.empty() || this->curr_img.empty()))
            {
                cv::Mat feat_img;
                cv::drawMatches(this->prev_img, this->prev_img_kps, this->curr_img, curr_img_kps, good_matches, feat_img, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                this->feature_map_pub->publish(OpenCVConversions::toRosImage(feat_img));
            }

            if (this->depth_complete)
            {
                this->motionEstimation();
            }
            
            this->prev_img_desc = curr_img_desc;
            this->prev_img_kps = curr_img_kps;

            this->feat_complete = true;
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

        // Undistort images
        cv::Mat rect_limg = this->undistortImage(OpenCVConversions::toCvImage(lcam_msg), this->lcam_intrinsics);
        cv::Mat rect_rimg = this->undistortImage(OpenCVConversions::toCvImage(rcam_msg), this->rcam_intrinsics);

        // Update image variables
        this->prev_img = this->curr_img;
        this->curr_img = rect_limg;

        // Create and send requests
        sensor_msgs::msg::Image rect_limg_msg = OpenCVConversions::toRosImage(rect_limg);
        sensor_msgs::msg::Image rect_rimg_msg = OpenCVConversions::toRosImage(rect_rimg);

        // Depth estimation request
        auto depth_estimator_request = std::make_shared<vio_msgs::srv::DepthEstimator::Request>();
        depth_estimator_request->left_image = rect_limg_msg;
        depth_estimator_request->right_image = rect_rimg_msg;
        waitForService(this->depth_estimator_client);
        this->depth_estimator_client->async_send_request(depth_estimator_request, std::bind(&StereoVONode::depth_callback, this, std::placeholders::_1));
        this->depth_complete = false;
        RCLCPP_INFO(this->get_logger(), "Depth estimation request sent.");

        // Feature matching request
        auto feature_extractor_request = std::make_shared<vio_msgs::srv::FeatureExtractor::Request>();
        feature_extractor_request->curr_img = rect_limg_msg;
        feature_extractor_request->prev_img_desc = OpenCVConversions::toRosImage(this->prev_img_desc);
        waitForService(this->feature_extractor_client);
        this->feature_extractor_client->async_send_request(feature_extractor_request, std::bind(&StereoVONode::feature_callback, this, std::placeholders::_1));
        this->feat_complete = false;
        RCLCPP_INFO(this->get_logger(), "Feature extraction request sent.");
    }   

public:

    StereoVONode(void) : Node("stereo_vo")
    {
        RCLCPP_INFO(this->get_logger(), "Starting stereo visual odometry...");

        // Parse config
        std::string stereo_img_topic, odometry_topic;
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
        vo_config["odom_topic"] >> odometry_topic;
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

        this->odometry_pub = this->create_publisher<geometry_msgs::msg::Odometry>(odometry_topic, 10);

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
