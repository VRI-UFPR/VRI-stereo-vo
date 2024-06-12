#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/feature_extractor.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/cudafeatures2d.hpp>

class FeatureExtractorServer : public rclcpp::Node
{
private:

    rclcpp::Service<vio_msgs::srv::FeatureExtractor>::SharedPtr feature_server;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_publisher;

    cv::Ptr<cv::cuda::ORB> orb_detector;

    void feature_callback(const std::shared_ptr<vio_msgs::srv::FeatureExtractor::Request> request,
                        std::shared_ptr<vio_msgs::srv::FeatureExtractor::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature extraction request. Size: " 
            << request->image.width << "x" << request->image.height);
            
        cv::Mat img = cv_bridge::toCvCopy(request->image, "mono8")->image;

        auto estimation_start = std::chrono::high_resolution_clock::now();
        
        // Upload to gpu
        cv::cuda::GpuMat cuda_img;
        cuda_img.upload(img);

        cv::cuda::GpuMat cuda_keypoints, cuda_descriptors;
        this->orb_detector->detectAndComputeAsync(cuda_img, cv::cuda::GpuMat(), cuda_keypoints, cuda_descriptors);

        // Download from gpu
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        this->orb_detector->convert(cuda_keypoints, keypoints);
        cuda_descriptors.download(descriptors);

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Feature extraction time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());

        // response->key_points = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", keypoints).toImageMsg());
        response->descriptors = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", descriptors).toImageMsg());

        // Draw keypoints and publish
        cv::Mat img_with_keypoints;
        cv::drawKeypoints(img, keypoints, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        this->feature_publisher->publish(*(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img_with_keypoints).toImageMsg()));

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
        this->feature_server = this->create_service<vio_msgs::srv::FeatureExtractor>(feature_extractor_service, 
            std::bind(&FeatureExtractorServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize publisher
        this->feature_publisher = this->create_publisher<sensor_msgs::msg::Image>("/stereo_vo/feature_image", 10);

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