#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "opencv_conversions.hpp"

#include <string>
#include <chrono>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "depth_estimator.hpp"

class DepthEstimatorServer : public rclcpp::Node
{
private:

    rclcpp::Service<vio_msgs::srv::DepthEstimator>::SharedPtr depth_server;

    std::shared_ptr<DepthEstimator> depth_estimator;

    bool estimate_depth = true;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscriber;
    std::deque<sensor_msgs::msg::Image> depth_buffer;

    void depth_callback(const std::shared_ptr<vio_msgs::srv::DepthEstimator::Request> request,
                        std::shared_ptr<vio_msgs::srv::DepthEstimator::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received depth estimation request. Size: " 
            << request->left_image.width << "x" << request->left_image.height << ", " 
            << request->right_image.width << "x" << request->right_image.height);
        auto estimation_start = std::chrono::high_resolution_clock::now();
        
        if (this->estimate_depth)
        {
            cv::Mat img_left = OpenCVConversions::toCvImage(request->left_image, "mono8");
            cv::Mat img_right = OpenCVConversions::toCvImage(request->right_image, "mono8");
            
            cv::Mat depth_map = this->depth_estimator->compute(img_left, img_right);

            response->depth_map = OpenCVConversions::toRosImage(depth_map, "mono8");
        } 
        else {
            // Support for datasets with ground truth depth (i.e. UFPR_MAP)
            // Search buffer for closest timestamp
            rclcpp::Time req_time = request->left_image.header.stamp;

            // Find closest timestamp
            sensor_msgs::msg::Image depth_img_cv = OpenCVConversions::toRosImage(cv::Mat(request->left_image.height, request->left_image.width, CV_8UC1, cv::Scalar(0)), "mono8");
            uint64_t min_diff = UINT64_MAX;
            for (auto img : this->depth_buffer)
            {
                rclcpp::Time img_time = img.header.stamp;
                double diff = std::abs(req_time.nanoseconds() - img_time.nanoseconds());
                if (diff < min_diff)
                {
                    min_diff = diff;
                    depth_img_cv = sensor_msgs::msg::Image(img);
                }
            }

            response->depth_map = depth_img_cv;
        }

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Depth estimation time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());
    }

    void depth_sub_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        this->depth_buffer.push_front(*msg);
        if (this->depth_buffer.size() > 150)
        {
            this->depth_buffer.pop_back();
        }
    }

public:
    DepthEstimatorServer(): Node("depth_estimator_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting depth estimator server...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);

        YAML::Node main_config = YAML::LoadFile(config_file); 
        std::string preset_path = "/workspace/config/" + main_config["preset"].as<std::string>() + ".yaml";
        YAML::Node preset_config = YAML::LoadFile(preset_path);
        
        // Parse parameters
        YAML::Node de_config = main_config["depth_estimator_params"];

        std::string depth_estimator_service = preset_config["depth_estimation_service"].as<std::string>();
        std::string lcam_topic = preset_config["left_cam"]["topic"].as<std::string>();
        std::string rcam_topic = preset_config["right_cam"]["topic"].as<std::string>();
        std::string lcam_intrinsics_file = preset_config["left_cam"]["intrinsics_file"].as<std::string>();
        std::string rcam_intrinsics_file = preset_config["right_cam"]["intrinsics_file"].as<std::string>();
        cv::Size img_size(preset_config["im_size"]["width"].as<int>(), preset_config["im_size"]["height"].as<int>());
        this->estimate_depth = preset_config["depth"].as<bool>();

        // Initialize service
        this->depth_server = this->create_service<vio_msgs::srv::DepthEstimator>(depth_estimator_service, 
            std::bind(&DepthEstimatorServer::depth_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize depth estimator
        if (this->estimate_depth)
        {
            this->depth_estimator = std::make_shared<DepthEstimator>(de_config, lcam_intrinsics_file, rcam_intrinsics_file, img_size);
        } 
        else {
            // Support for datasets with ground truth depth (i.e. UFPR_MAP)
            std::string depth_topic = preset_config["depth_viz"].as<std::string>();
            RCLCPP_INFO_STREAM(this->get_logger(), "Will use ground truth depth from topic: " << depth_topic);

            this->depth_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
                depth_topic, 10, std::bind(&DepthEstimatorServer::depth_sub_callback, this, std::placeholders::_1));
        }
        
        RCLCPP_INFO(this->get_logger(), "Depth estimator server started.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthEstimatorServer>());
    rclcpp::shutdown();
    return 0;
}