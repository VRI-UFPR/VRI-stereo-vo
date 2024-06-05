#include "rclcpp/rclcpp.hpp"

#include "stereo_vo/srv/feature_extractor.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/cudafeatures2d.hpp>

class FeatureExtractorServer : public rclcpp::Node
{
private:

    rclcpp::Service<stereo_vo::srv::FeatureExtractor>::SharedPtr feature_server;

    cv::Ptr<cv::cuda::ORB> orb_detector;

    void feature_callback(const std::shared_ptr<stereo_vo::srv::FeatureExtractor::Request> request,
                        std::shared_ptr<stereo_vo::srv::FeatureExtractor::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature extraction request. Size: " 
            << request->image.width << "x" << request->image.height);
            
        cv::Mat img = cv_bridge::toCvCopy(request->image, "mono8")->image;

        // Upload to gpu
        cv::cuda::GpuMat cuda_img;
        cuda_img.upload(img);

        cv::cuda::GpuMat cuda_keypoints;
        cv::cuda::GpuMat cuda_descriptors;
        this->orb_detector->detectAndComputeAsync(cuda_img, cv::cuda::GpuMat(), cuda_keypoints, cuda_descriptors);

        // Download from gpu
        cv::Mat keypoints, descriptors;
        cuda_keypoints.download(keypoints);
        cuda_descriptors.download(descriptors);

        response->key_points = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", keypoints).toImageMsg());
        response->descriptors = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", descriptors).toImageMsg());

        RCLCPP_INFO(this->get_logger(), "Feature extraction completed.");
    }

public:

    FeatureExtractorServer(): Node("feature_extractor_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting feature extractor server...");

        // Parse parameters
        std::string feature_extractor_service; 
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["feature_extractor_service"] >> feature_extractor_service;
        fs.release();

        // Initialize service
        this->feature_server = this->create_service<stereo_vo::srv::FeatureExtractor>(feature_extractor_service, 
            std::bind(&FeatureExtractorServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        this->orb_detector = cv::cuda::ORB::create();

        RCLCPP_INFO(this->get_logger(), "Feature extractor server started.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FeatureExtractorServer>());
    rclcpp::shutdown();
    return 0;
}