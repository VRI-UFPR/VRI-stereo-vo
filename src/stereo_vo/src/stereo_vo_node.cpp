#include <string>
#include <chrono>
#include <deque>
#include <tuple>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "vio_msgs/srv/feature_extractor.hpp"
#include "vio_msgs/msg/d_matches.hpp"
#include "vio_msgs/msg/key_points.hpp"

#include "opencv_conversions.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

class StereoVONode : public rclcpp::Node
{
private:

    // Stereo vo state variables
    cv::Mat curr_img = cv::Mat(), prev_img = cv::Mat();
    cv::Mat prev_img_desc = cv::Mat();
    cv::Mat curr_depth_map = cv::Mat();
    std::vector<cv::KeyPoint> prev_img_kps, curr_img_kps;
    std::vector<cv::DMatch> good_matches;
    Eigen::Matrix4d curr_pose = Eigen::Matrix4d::Identity();

    // Camera intrinsics
    OpenCVConversions::CameraIntrinsics lcam_intrinsics, rcam_intrinsics;
    
    // Image subscriber
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr lcam_sub;
    sensor_msgs::msg::Image next_img;

    // Depth subscriber
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscriber;
    std::deque<sensor_msgs::msg::Image> depth_buffer;
    size_t depth_buffer_size = 5;

    // Visualization publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_map_pub;
    bool enable_viz = false;

    // Clients
    rclcpp::Client<vio_msgs::srv::FeatureExtractor>::SharedPtr feature_extractor_client;
    vio_msgs::srv::FeatureExtractor::Response::SharedPtr feature_response;

    // Pose publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub;

    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor;

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

    vio_msgs::srv::FeatureExtractor::Response::SharedPtr featureExtractionRequest(const cv::Mat &curr_img)
    {
        sensor_msgs::msg::Image curr_img_msg = OpenCVConversions::toRosImage(curr_img);

        // Feature matching request
        auto feature_extractor_request = std::make_shared<vio_msgs::srv::FeatureExtractor::Request>();
        feature_extractor_request->curr_img = curr_img_msg;
        feature_extractor_request->prev_img_desc = OpenCVConversions::toRosImage(this->prev_img_desc);

        // Send request
        waitForService(this->feature_extractor_client);
        auto future = this->feature_extractor_client->async_send_request(feature_extractor_request);
        RCLCPP_INFO(this->get_logger(), "Feature extraction request sent.");
        
        // Wait for response
        if (this->executor->spin_until_future_complete(future) == rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_INFO(this->get_logger(), "Feature extraction response received.");
            return future.get();
        } else {
            RCLCPP_ERROR(this->get_logger(), "Feature extraction request failed.");
            return nullptr;
        }
    }

    void publishOdometry(const Eigen::Matrix4d &pose)
    {
        nav_msgs::msg::Odometry odometry_msg;
        odometry_msg.header.stamp = this->now();

        // Convert to meters
        odometry_msg.pose.pose.position.x = pose(0, 3);
        odometry_msg.pose.pose.position.y = pose(1, 3);
        odometry_msg.pose.pose.position.z = pose(2, 3);

        Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
        odometry_msg.pose.pose.orientation.x = q.x();
        odometry_msg.pose.pose.orientation.y = q.y();
        odometry_msg.pose.pose.orientation.z = q.z();
        odometry_msg.pose.pose.orientation.w = q.w();


        RCLCPP_INFO_STREAM(this->get_logger(), "Estimated pose: " << std::endl << this->curr_pose.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")));

        this->odometry_pub->publish(odometry_msg);
    }

    void motionEstimation(void)
    {
        double cx = this->lcam_intrinsics.cx();
        double cy = this->lcam_intrinsics.cy();
        double fx = this->lcam_intrinsics.fx();
        double fy = this->lcam_intrinsics.fy();
        
        double baseline = abs(this->lcam_intrinsics.tlVector().at<double>(0) - this->rcam_intrinsics.tlVector().at<double>(0));
        double depth_scale = fx * baseline;

        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;
        for (cv::DMatch &m : this->good_matches)
        {
            // Calculate previous image 3D point
            cv::Point2d p = this->prev_img_kps[m.trainIdx].pt;
            // Needs to use uchar because depth map is mono8 image
            double z = static_cast<double>(this->curr_depth_map.at<uchar>(p.y, p.x));
            if (z == 0.0)
            {
                continue;
            }

            // z = depth_scale / z;
            double x = (p.x - cx) * z / fx;
            double y = (p.y - cy) * z / fy;

            pts_3d.push_back(cv::Point3f(x, y, z));
            
            // Get current image 2D point
            pts_2d.push_back(this->curr_img_kps[m.queryIdx].pt);

            RCLCPP_INFO_STREAM(this->get_logger(), "DS: " << depth_scale << " Points: " << x << ", " << y << ", " << z << " | " << p.x << ", " << p.y << " | " << this->curr_img_kps[m.queryIdx].pt.x << ", " << this->curr_img_kps[m.queryIdx].pt.y);
        }

        // Solve PnP
        if ((pts_3d.size() < 6) || (pts_3d.size() != pts_2d.size()))
        {
            RCLCPP_WARN(this->get_logger(), "Insufficient points for PnP estimation.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Estimating motion. Points: %ld", pts_3d.size());
        cv::Mat rvec, tvec, R;        
        try {
            cv::solvePnPRansac(pts_3d, pts_2d, this->lcam_intrinsics.cameraMatrix(), this->lcam_intrinsics.distCoeffs(), rvec, tvec);
        } catch (cv::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "PnP failed: %s", e.what());
            return;
        }
        cv::Rodrigues(rvec, R);

        // Convert to Transformation matrix and update current pose
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R.ptr<double>());
        T(0, 3) = tvec.at<double>(0);
        T(1, 3) = tvec.at<double>(1);
        T(2, 3) = tvec.at<double>(2);

        this->curr_pose *= T;

        this->publishOdometry(this->curr_pose);
    }

    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg)
    {
        this->depth_buffer.push_front(*depth_msg);

        if (this->depth_buffer.size() > this->depth_buffer_size)
        {
            this->depth_buffer.pop_back();
        }
    }

    void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr &lcam_msg)
    {
        this->next_img = *lcam_msg;
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
        
        this->feature_map_pub->publish(OpenCVConversions::toRosImage(feat_img, "bgr8"));
    }

public:

    StereoVONode(void) : Node("stereo_vo")
    {
        RCLCPP_INFO(this->get_logger(), "Starting stereo visual odometry...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config_imx.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);
        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];

        // Parse config
        std::string lcam_topic = vo_config["left_cam"]["topic"];
        std::string lcam_intrinsics_file = vo_config["left_cam"]["intrinsics_file"];
        std::string depth_topic = vo_config["depth_estimator_params"]["topic"];
        std::string feature_extractor_service = vo_config["feature_extractor_service"];
        this->enable_viz = vo_config["debug_visualization"].real();
        std::string feature_viz_topic = vo_config["feature_viz"];
        std::string odometry_topic = vo_config["odom_topic"];
        fs.release();

        // Load camera intrinsics
        this->lcam_intrinsics = OpenCVConversions::CameraIntrinsics(lcam_intrinsics_file);
        this->rcam_intrinsics = OpenCVConversions::CameraIntrinsics(lcam_intrinsics_file);

        // Initialize subscribers and client
        this->lcam_sub = this->create_subscription<sensor_msgs::msg::Image>(lcam_topic, 10, std::bind(&StereoVONode::img_callback, this, std::placeholders::_1));
        this->depth_subscriber = this->create_subscription<sensor_msgs::msg::Image>(depth_topic, 10, std::bind(&StereoVONode::depth_callback, this, std::placeholders::_1));
        this->feature_extractor_client = this->create_client<vio_msgs::srv::FeatureExtractor>(feature_extractor_service);

        // Initialize publishers
        if (this->enable_viz)
        {
            this->feature_map_pub = this->create_publisher<sensor_msgs::msg::Image>(feature_viz_topic, 10);
        }
        this->odometry_pub = this->create_publisher<nav_msgs::msg::Odometry>(odometry_topic, 10);

        RCLCPP_INFO(this->get_logger(), "Stereo visual odometry node started.");
    }

    void run(void)
    {
        this->executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        this->executor->add_node(get_node_base_interface());

        while(rclcpp::ok())
        {
            this->executor->spin_some(std::chrono::milliseconds(300));
            
            if (this->next_img.header.stamp == rclcpp::Time(0))
            {
                continue;
            }

            rclcpp::Time curr_img_stamp = this->next_img.header.stamp;

            // Undistort images
            cv::Mat rect_limg = this->lcam_intrinsics.undistortImage(OpenCVConversions::toCvImage(this->next_img));
            this->next_img.header.stamp = rclcpp::Time(0);

            // Feature extraction request
            auto feat_response = this->featureExtractionRequest(rect_limg);
            if (feat_response == nullptr)
            {
                continue;
            }

            // Update image variables
            this->prev_img = this->curr_img;
            this->curr_img = rect_limg;

            // Convert to OpenCV types
            cv::Mat curr_img_desc = OpenCVConversions::toCvImage(feat_response->curr_img_desc);
            this->curr_img_kps = OpenCVConversions::toCvKeyPoints(feat_response->curr_img_kps);
            this->good_matches = OpenCVConversions::toCvDMatches(feat_response->good_matches);

            // Publish feature visualization
            if (this->enable_viz && !(this->prev_img.empty() || this->curr_img.empty()) && (this->good_matches.size() > 0))
            {
                this->publishFeatureMap();
            }

            // Search for depth map
            if (this->depth_buffer.size() > 0)
            {
                bool depth_found = false;
                RCLCPP_INFO(this->get_logger(), "Searching for the closest depth timestamp: %ld", curr_img_stamp.nanoseconds());

                for (sensor_msgs::msg::Image depth_msg : this->depth_buffer)
                {
                    rclcpp::Time depth_time = depth_msg.header.stamp;

                    RCLCPP_INFO(this->get_logger(), "   %ld", depth_time.nanoseconds());
                    if (depth_time.nanoseconds() <= curr_img_stamp.nanoseconds())
                    {   
                        this->curr_depth_map = OpenCVConversions::toCvImage(depth_msg, "");
                        depth_found = true;
                        break;
                    }
                }

                // Run motion estimation
                if (depth_found)
                {
                    this->motionEstimation();
                } else {
                    RCLCPP_WARN(this->get_logger(), "Could not find matching depth timestamp.");
                }
                
            } else {
                RCLCPP_WARN(this->get_logger(), "Depth buffer is empty.");
            }

            this->prev_img_desc = curr_img_desc;
            this->prev_img_kps = curr_img_kps;
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
