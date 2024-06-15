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

    cv::Ptr<cv::cuda::ORB> orb_detector;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

    void featureExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
    {
        // Upload to gpu
        cv::cuda::GpuMat cuda_img;
        cuda_img.upload(img);

        cv::cuda::GpuMat cuda_keypoints, cuda_descriptors;
        this->orb_detector->detectAndComputeAsync(cuda_img, cv::cuda::GpuMat(), cuda_keypoints, cuda_descriptors);

        // Download from gpu
        this->orb_detector->convert(cuda_keypoints, keypoints);
        cuda_descriptors.download(descriptors);
    }

    void featureMatch(const cv::Mat &curr_img_desc, const cv::Mat &prev_img_desc, std::vector<cv::DMatch> &good_matches)
    {
        // Upload to gpu
        cv::cuda::GpuMat cuda_curr_desc, cuda_prev_desc;
        cuda_curr_desc.upload(curr_img_desc);
        cuda_prev_desc.upload(prev_img_desc);

        std::vector<std::vector<cv::DMatch>> matches;
        this->matcher->knnMatch(cuda_curr_desc, cuda_prev_desc, matches, 2);

        // As per Lowe's ratio test
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance < 0.8 * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
    }

    void feature_callback(const std::shared_ptr<vio_msgs::srv::FeatureExtractor::Request> request,
                        std::shared_ptr<vio_msgs::srv::FeatureExtractor::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature extraction request. Size: " 
            << request->curr_img.width << "x" << request->curr_img.height);
        auto estimation_start = std::chrono::high_resolution_clock::now();
        
        // Convert data from request
        cv::Mat curr_img = cv_bridge::toCvCopy(request->curr_img, "mono8")->image;
        cv::Mat prev_img_desc = cv_bridge::toCvCopy(request->prev_img_desc, "mono8")->image;

        // Extract features from current image
        cv::Mat curr_img_desc;
        std::vector<cv::KeyPoint> curr_img_kp;
        this->featureExtract(curr_img, curr_img_kp, curr_img_desc);

        // Match features between current and previous image
        std::vector<cv::DMatch> good_matches;
        this->featureMatch(curr_img_desc, prev_img_desc, good_matches);

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_DEBUG(this->get_logger(), "Feature extraction time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());

        // Convert to ros message
        response->curr_img_desc = *(cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", curr_img_desc).toImageMsg());
        for (size_t i = 0; i < good_matches.size(); i++)
        {
            response->curr_img_points.data.push_back(good_matches[i].queryIdx);
            response->prev_img_points.data.push_back(good_matches[i].trainIdx);
            response->distances.data.push_back(good_matches[i].distance);
        }
        for (size_t i = 0; i < curr_img_kp.size(); i++)
        {
            response->curr_img_keypoints.data.push_back(curr_img_kp[i].pt.x);
            response->curr_img_keypoints.data.push_back(curr_img_kp[i].pt.y);
        }

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

        // Initialize feature extractor
        this->orb_detector = cv::cuda::ORB::create();
        this->matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        this->matcher->train();

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