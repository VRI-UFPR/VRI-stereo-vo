#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/feature_matcher.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/cudafeatures2d.hpp>

class FeatureMatcherServer : public rclcpp::Node
{
private:

    rclcpp::Service<vio_msgs::srv::FeatureMatcher>::SharedPtr feature_server;

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

    void feature_callback(const std::shared_ptr<vio_msgs::srv::FeatureMatcher::Request> request,
                        std::shared_ptr<vio_msgs::srv::FeatureMatcher::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature matching request. Size: " 
            << request->current_image_desc.width << "x" << request->current_image_desc.height << ", " 
            << request->previous_image_desc.width << "x" << request->previous_image_desc.height);
            
        cv::Mat current_image_desc = cv_bridge::toCvCopy(request->current_image_desc, "mono8")->image;
        cv::Mat previous_image_desc = cv_bridge::toCvCopy(request->previous_image_desc, "mono8")->image;

        // Upload to gpu
        cv::cuda::GpuMat cuda_current_desc, cuda_previous_desc;
        cuda_current_desc.upload(current_image_desc);
        cuda_previous_desc.upload(previous_image_desc);

        std::vector<std::vector<cv::DMatch>> matches;
        this->matcher->knnMatch(cuda_current_desc, cuda_previous_desc, matches, 2);

        // As per Lowe's ratio test
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance < 0.8 * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }

        // Convert to ros message
        for (size_t i = 0; i < good_matches.size(); i++)
        {
            response->curr_img_points.data.push_back(good_matches[i].queryIdx);
            response->prev_img_points.data.push_back(good_matches[i].trainIdx);
            response->distances.data.push_back(good_matches[i].distance);
        }
    }

public:

    FeatureMatcherServer(): Node("feature_matcher_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting feature matcher server...");

        // Parse parameters
        std::string feature_matcher_service; 
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["feature_matcher_service"] >> feature_matcher_service;
        fs.release();

        // Initialize service
        this->feature_server = this->create_service<vio_msgs::srv::FeatureMatcher>(feature_matcher_service, 
            std::bind(&FeatureMatcherServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize matcher
        this->matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        this->matcher->train();

        RCLCPP_INFO(this->get_logger(), "Feature matcher server started.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FeatureMatcherServer>());
    rclcpp::shutdown();
    return 0;
}