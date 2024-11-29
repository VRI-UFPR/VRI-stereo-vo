#include <string>
#include <chrono>
#include <deque>
#include <tuple>
#include <future>
#include <set>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "opencv_conversions.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "depth_estimator.hpp"
#include "feature_matcher.hpp"

class StereoVONode : public rclcpp::Node
{
private:

    // Stereo vo state variables
    cv::Mat curr_img = cv::Mat(), prev_img = cv::Mat();
    cv::Mat prev_img_desc = cv::Mat();
    cv::Mat curr_depth_map = cv::Mat();
    cv::Mat prev_depth_map = cv::Mat();
    std::vector<cv::KeyPoint> prev_img_kps, curr_img_kps;
    std::vector<cv::DMatch> good_matches;
    Eigen::Matrix4d curr_pose = Eigen::Matrix4d::Identity();
    cv::Size img_size;
    bool skip_frame = false;

    // Feature extractor
    std::shared_ptr<FeatureMatcher> feature_extractor;
    // Depth estimator
    std::shared_ptr<DepthEstimator> depth_estimator;

    // Image buffer
    size_t buffer_size = 1;
    std::deque<std::tuple<int, sensor_msgs::msg::Image, sensor_msgs::msg::Image>> image_buffer;

    // Reprojection error
    double reprojection_threshold;
    std::vector<double> reproj_erros;
    long total_reprojection_errors = 0;
    double total_reprojection_error = 0;
    std::set<double> reprojection_errors;

    // Camera intrinsics
    OpenCVConversions::CameraIntrinsics lcam_intrinsics, rcam_intrinsics;
    double baseline = 0.012;
    bool undistort = false;
    
    // Subscribers and time synchronizer
    message_filters::Subscriber<sensor_msgs::msg::Image> lcam_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> rcam_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cam_sync;

    int last_pose_id = -1;

    // Visualization publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_map_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_map_pub;
    bool enable_viz = false;
    bool estimate_depth = true;
    std::vector<double> odom_scale = {1.0, 1.0, 1.0};

    // Pose publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub;

    // Output file for kitti dataset
    std::ofstream kitti_file;

    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor;

    long total_motion_estimations = 0;
    long total_motion_estimation_time = 0;
    long total_pose_estimations = 0;
    long total_pose_estimation_time = 0;

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

    void publishOdometry(const Eigen::Matrix4d &pose, int pose_id)
    {
        nav_msgs::msg::Odometry odometry_msg;
        odometry_msg.header.stamp = this->now();

        this->kitti_file << pose_id<< " " << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << " " << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << " " << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << std::endl;

        RCLCPP_INFO_STREAM(this->get_logger(), "Current position: " << pose(0, 3) << " " << pose(1, 3) << " " << pose(2, 3));

        odometry_msg.pose.pose.position.x = pose(0, 3) * this->odom_scale[0];
        odometry_msg.pose.pose.position.y = pose(1, 3) * this->odom_scale[1];
        odometry_msg.pose.pose.position.z = pose(2, 3) * this->odom_scale[2];

        Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
        odometry_msg.pose.pose.orientation.x = q.x();
        odometry_msg.pose.pose.orientation.y = q.y();
        odometry_msg.pose.pose.orientation.z = q.z();
        odometry_msg.pose.pose.orientation.w = q.w();

        this->odometry_pub->publish(odometry_msg);
    }

    bool motionEstimation(int pose_id)
    {
        auto start = std::chrono::high_resolution_clock::now();

        double cx = this->lcam_intrinsics.cx();
        double cy = this->lcam_intrinsics.cy();
        double fx = this->lcam_intrinsics.fx();
        double fy = this->lcam_intrinsics.fy();
        
        double depth_scale = fx * this->baseline;
        // Eigen::Map<Eigen::Matrix3d> K(this->lcam_intrinsics.cameraMatrix().ptr<double>(), 3, 3);
        
        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;
        for (cv::DMatch &m : this->good_matches)
        {
            // Calculate previous image 3D point
            cv::Point2d p = this->prev_img_kps[m.queryIdx].pt;
            // Needs to use uchar because depth map is mono8 image
            double z = static_cast<double>(this->prev_depth_map.at<uchar>(p.y, p.x));
            if (z <= 0)
            {
                continue;
            }

            z = depth_scale / z;
            double x = (p.x - cx) * z / fx;
            double y = (p.y - cy) * z / fy;

            pts_3d.push_back(cv::Point3d(x, y, z));
            
            // Get current image 2D point
            pts_2d.push_back(this->curr_img_kps[m.trainIdx].pt);

        }
        
        // Solve PnP
        if (pts_3d.size() < 6)
        {
            RCLCPP_WARN(this->get_logger(), "Insufficient points for PnP estimation.");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Estimating motion. Points: %ld", pts_3d.size());
        cv::Mat rvec, tvec, R;
        bool estm_intrinsic = false;
        int mode = cv::SOLVEPNP_AP3P;
        for (int i = 0; i < 1; i++)
        {
            try {
                cv::solvePnPRansac(pts_3d, pts_2d, this->lcam_intrinsics.cameraMatrix(), this->lcam_intrinsics.distCoeffs(), rvec, tvec, estm_intrinsic, 200, 5.0, 0.99, cv::noArray(), mode);
            } catch (cv::Exception &e) {
                RCLCPP_ERROR(this->get_logger(), "PnP failed: %s", e.what());
                return false;
            }

            if (!estm_intrinsic)
            {
                estm_intrinsic = true;
                mode = cv::SOLVEPNP_ITERATIVE;
            }
        }

        cv::Rodrigues(rvec, R);

        // Filter usign reprojection error
        std::vector<cv::Point2d> projected_pts;
        cv::projectPoints(pts_3d, rvec, tvec, this->lcam_intrinsics.cameraMatrix(), this->lcam_intrinsics.distCoeffs(), projected_pts);

        // Average error
        double error = 0.0;
        for (size_t i = 0; i < pts_2d.size(); i++)
        {
            error += cv::norm(pts_2d[i] - projected_pts[i]);
        }
        error /= pts_2d.size();

        RCLCPP_INFO_STREAM(this->get_logger(), "Reprojection error: " << error);
        this->total_reprojection_errors++;
        this->total_reprojection_error += error;
        this->reprojection_errors.insert(error);
        std::set<double>::iterator it = this->reprojection_errors.begin();
        std::advance(it, this->reprojection_errors.size() / 2);
        double median = *it;

        // Convert to Transformation matrix
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R.ptr<double>());
        T(0, 3) = tvec.at<double>(0);
        T(1, 3) = tvec.at<double>(1);
        T(2, 3) = tvec.at<double>(2);

        RCLCPP_INFO_STREAM(this->get_logger(), "Estimated transformation: " << std::endl << T.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")));

        RCLCPP_INFO_STREAM(this->get_logger(), "Reprojection error: " << error << " - Average: " << this->total_reprojection_error / this->total_reprojection_errors << " - Median: " << median);  

        if (error > this->reprojection_threshold)
        {
            RCLCPP_WARN(this->get_logger(), "Motion rejected due to high reprojection error.");
            return false;
        }

        // Only accept pose if dominant motion is foward
        // if ((T(2, 3) > 0) || (abs(T(2, 3)) < abs(T(0, 3))) || (abs(T(2, 3)) < abs(T(1, 3))))
        // {
        //     // RCLCPP_WARN(this->get_logger(), "Motion rejected.");        
        //     // return;
        // }

        // Update pose and publish odometry
        this->curr_pose *= T.inverse();

        Eigen::Matrix4d pose_rh = this->curr_pose;

        // Correct for right-handed coordinate system
        pose_rh(0, 1) = -pose_rh(0, 1);
        pose_rh(0, 2) = -pose_rh(0, 2);
        pose_rh(0, 3) = -pose_rh(0, 3);
        pose_rh(1, 0) = -pose_rh(1, 0);
        pose_rh(2, 0) = -pose_rh(2, 0);

        this->publishOdometry(pose_rh, pose_id);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        this->total_motion_estimations++;
        this->total_motion_estimation_time += duration.count();
        RCLCPP_INFO_STREAM(this->get_logger(), "Motion estimation time: " << duration.count() << "ms" << " - Average: " << (double)this->total_motion_estimation_time / this->total_motion_estimations);

        return true;
    }
    
    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr &lcam_msg, const sensor_msgs::msg::Image::ConstSharedPtr &rcam_msg)
    {
        this->last_pose_id++;

        this->image_buffer.push_back(std::make_tuple(this->last_pose_id, *lcam_msg, *rcam_msg));
        if (this->image_buffer.size() > this->buffer_size)
        {
            this->image_buffer.pop_front();
        }
        this->skip_frame = true;
    }   

    void publishFeatureMap(void)
    {
        cv::Mat feat_img;
        try
        {
            cv::drawMatches(this->prev_img, this->prev_img_kps, this->curr_img, this->curr_img_kps, this->good_matches, feat_img, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        } catch (cv::Exception &e) {
            RCLCPP_WARN(this->get_logger(), "Draw matches failed: %s", e.what());
        }
        
        this->feature_map_pub->publish(OpenCVConversions::toRosImage(feat_img));
    }

public:

    StereoVONode(void) : Node("stereo_vo")
    {
        RCLCPP_INFO(this->get_logger(), "Starting stereo visual odometry...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);

        YAML::Node main_config = YAML::LoadFile(config_file); 
        std::string preset_path = "/workspace/config/" + main_config["preset"].as<std::string>() + ".yaml";
        YAML::Node preset_config = YAML::LoadFile(preset_path);
        
        // Parse config
        std::string lcam_topic = preset_config["left_cam"]["topic"].as<std::string>();
        std::string rcam_topic = preset_config["right_cam"]["topic"].as<std::string>();
        std::string lcam_intrinsics_file = preset_config["left_cam"]["intrinsics_file"].as<std::string>();
        std::string rcam_intrinsics_file = preset_config["right_cam"]["intrinsics_file"].as<std::string>();
        std::string feature_viz_topic = preset_config["feature_viz"].as<std::string>();
        std::string depth_viz_topic = preset_config["depth_viz"].as<std::string>();
        std::string odometry_topic = preset_config["vo_odom_topic"].as<std::string>();
        this->undistort = preset_config["undistort"].as<bool>();
        this->baseline = preset_config["baseline"].as<double>();
        this->estimate_depth = preset_config["depth"].as<bool>();
        this->odom_scale = preset_config["odom_scale"].as<std::vector<double>>();
        this->img_size = cv::Size(preset_config["im_size"]["width"].as<int>(), preset_config["im_size"]["height"].as<int>());

        YAML::Node stereo_vo_config = main_config["stereo_vo"];
        this->enable_viz = stereo_vo_config["debug_visualization"].as<bool>();
        this->reprojection_threshold = stereo_vo_config["reprojection_threshold"].as<double>();
        this->buffer_size = stereo_vo_config["buffer_size"].as<size_t>();

        // Load camera intrinsics
        this->lcam_intrinsics = OpenCVConversions::CameraIntrinsics(lcam_intrinsics_file);
        this->rcam_intrinsics = OpenCVConversions::CameraIntrinsics(rcam_intrinsics_file);

        // Initialize subscribers and time synchronizer
        this->lcam_sub.subscribe(this, lcam_topic);
        this->rcam_sub.subscribe(this, rcam_topic);        
        this->cam_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), lcam_sub, rcam_sub);
        this->cam_sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.3));
        this->cam_sync->registerCallback(std::bind(&StereoVONode::stereo_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize services
        YAML::Node feature_matcher_config = main_config["feature_matcher"];
        YAML::Node depth_estimator_config = main_config["depth_estimator"];
        this->feature_extractor = std::make_shared<FeatureMatcher>(feature_matcher_config);
        this->depth_estimator = std::make_shared<DepthEstimator>(depth_estimator_config, this->img_size);

        // Initialize publishers
        if (this->enable_viz)
        {
            this->feature_map_pub = this->create_publisher<sensor_msgs::msg::Image>(feature_viz_topic, 10);
            this->depth_map_pub = this->create_publisher<sensor_msgs::msg::Image>(depth_viz_topic, 10);
        }
        this->odometry_pub = this->create_publisher<nav_msgs::msg::Odometry>(odometry_topic, 10);

        // Initialize output file
        std::string output_file = "/workspace/Data/kitti_odom.txt";
        this->kitti_file.open(output_file);
        if (!this->kitti_file.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open output file: %s", output_file.c_str());
        }

        RCLCPP_INFO(this->get_logger(), "Stereo visual odometry node started.");
    }

    ~StereoVONode(void)
    {
        this->kitti_file.close();
    }

    void run(void)
    {
        this->executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        this->executor->add_node(get_node_base_interface());

        while(rclcpp::ok())
        {
            this->executor->spin_some(std::chrono::milliseconds(10));
            
            if (this->image_buffer.size() < 1)
            {
                continue;
            }

            int curr_id = std::get<0>(this->image_buffer.front());
            sensor_msgs::msg::Image next_limg = std::get<1>(this->image_buffer.front());
            sensor_msgs::msg::Image next_rimg = std::get<2>(this->image_buffer.front());
            this->image_buffer.pop_front();

            auto pose_estimation_start = std::chrono::high_resolution_clock::now();

            rclcpp::Time curr_img_stamp = next_limg.header.stamp;

            // Undistort images
            cv::Mat rect_limg;
            cv::Mat rect_rimg;
            if (this->undistort)
            {
               rect_limg = this->lcam_intrinsics.undistortImage(OpenCVConversions::toCvImage(next_limg));
               rect_rimg = this->rcam_intrinsics.undistortImage(OpenCVConversions::toCvImage(next_rimg));
            } else {
                rect_limg = OpenCVConversions::toCvImage(next_limg);
                rect_rimg = OpenCVConversions::toCvImage(next_rimg);
            }

            // Update image variables
            this->prev_img = this->curr_img;
            this->curr_img = rect_limg;

            // Send requests
            std::future<std::shared_ptr<DepthEstimator::DepthResponse>> depth_future = std::async(std::launch::async, [&](){
                return this->depth_estimator->compute(rect_limg, rect_rimg);
            });
            std::future<std::shared_ptr<FeatureMatcher::MatchResponse>> feature_future = std::async(std::launch::async, [&](){
                return this->feature_extractor->compute(rect_limg, this->prev_img_desc);
            });
            RCLCPP_INFO(this->get_logger(), "Feature and depth estimation requests sent.");

            // Spin while waiting for both responses
            while (depth_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready || feature_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            {
                this->executor->spin_some(std::chrono::milliseconds(10));
            }

            RCLCPP_INFO(this->get_logger(), "Feature and depth estimation completed.");

            std::shared_ptr<DepthEstimator::DepthResponse> depth_response = depth_future.get();
            std::shared_ptr<FeatureMatcher::MatchResponse> feature_response = feature_future.get();

            RCLCPP_INFO_STREAM(this->get_logger(), "Depth estimation time: " << depth_response->estimation_time << "ms - Average: " << depth_response->average_time);
            RCLCPP_INFO_STREAM(this->get_logger(), "Feature extraction time: " << feature_response->extraction_time << "ms - Matching time: " << feature_response->matching_time << "ms - Total: " << feature_response->total_time << "ms - Average: " << feature_response->average_time);

            cv::Mat curr_img_desc = feature_response->curr_img_desc;
            this->curr_img_kps = feature_response->curr_img_kp;
            this->good_matches = feature_response->good_matches;

            this->curr_depth_map = depth_response->disparity_map;

            // Publish feature visualization
            if (this->enable_viz && !(this->prev_img.empty() || this->curr_img.empty()) && (this->good_matches.size() > 0))
            {
                RCLCPP_INFO(this->get_logger(), "Publishing visualization.");
                this->publishFeatureMap();

                if (this->estimate_depth)
                {   
                    this->depth_map_pub->publish(OpenCVConversions::toRosImage(this->curr_depth_map));
                }
            }

            this->motionEstimation(curr_id);
        
            this->prev_img_desc = curr_img_desc;
            this->prev_img_kps = curr_img_kps;
            this->prev_depth_map = curr_depth_map;

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - pose_estimation_start);
            this->total_pose_estimations++;
            this->total_pose_estimation_time += duration.count();
            RCLCPP_INFO_STREAM(this->get_logger(), "Total pose estimation time: " << duration.count() << "ms - Average: " << (double)this->total_pose_estimation_time / this->total_pose_estimations << "ms\n\n\n");
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    StereoVONode vo_node;
    vo_node.run();
    rclcpp::shutdown();
    return 0;
}
