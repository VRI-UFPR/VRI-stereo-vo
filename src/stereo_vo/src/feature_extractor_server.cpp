#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/feature_extractor.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "opencv_conversions.hpp"

#include <string>
#include <vector>
#include <chrono>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#ifdef USE_CUDA
    #include <opencv2/cudafeatures2d.hpp>
#endif

#include <opencv2/features2d.hpp>

class FeatureExtractorServer : public rclcpp::Node
{
private:

    rclcpp::Service<vio_msgs::srv::FeatureExtractor>::SharedPtr feature_server;

    #ifdef USE_CUDA
        cv::Ptr<cv::cuda::ORB> cuda_orb_extractor;
    #else
        cv::Ptr<cv::Feature2D> feat_extractor;
    #endif

    cv::Ptr<cv::DescriptorMatcher> matcher;

    double distance_ratio = 0.3;

    void featureExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
    {
        #ifdef USE_CUDA
            // Upload to gpu
            cv::cuda::GpuMat cuda_img;
            cuda_img.upload(img);

            cv::cuda::GpuMat cuda_keypoints, cuda_descriptors;
            this->feat_extractor->detectAndComputeAsync(cuda_img, cv::cuda::GpuMat(), cuda_keypoints, cuda_descriptors);

            // Download from gpu
            this->feat_extractor->convert(cuda_keypoints, keypoints);
            cuda_descriptors.download(descriptors);
        #else
            this->feat_extractor->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        #endif
    }

    std::vector<cv::DMatch> ratioTest(std::vector<std::vector<cv::DMatch>> &matches)
    {
        std::vector<cv::DMatch> good_matches;

        // As per Lowe's ratio test
        for ( std::vector<std::vector<cv::DMatch> >::iterator matchIterator= matches.begin(); matchIterator!= matches.end(); ++matchIterator)
        {
            if (matchIterator->size() < 2)
                continue;

            if ((*matchIterator)[0].distance < this->distance_ratio * (*matchIterator)[1].distance)
            {
                good_matches.push_back((*matchIterator)[0]);
            }
        }

        return good_matches;
    }

    void symmetryTest(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2, std::vector<cv::DMatch>& symMatches)
    {
        symMatches.clear();
        for (std::vector<cv::DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
        {
            for (std::vector<cv::DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
            {
                if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator2).queryIdx == (*matchIterator1).trainIdx)
                {
                    symMatches.push_back(cv::DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
                    break;
                }
            }
        }
    }

    void featureMatch(const cv::Mat &curr_img_desc, const cv::Mat &prev_img_desc, std::vector<cv::DMatch> &good_matches)
    {
        std::vector<std::vector<cv::DMatch>> matches_pc, matches_cp;
        std::vector<cv::DMatch> matches_pc_good, matches_cp_good;

        // Match features
        if (curr_img_desc.empty() || prev_img_desc.empty())
        {
            return;
        }

        try
        {
            this->matcher->knnMatch(prev_img_desc, curr_img_desc, matches_pc, 2);
            matches_pc_good = this->ratioTest(matches_pc);

            this->matcher->knnMatch(curr_img_desc, prev_img_desc, matches_cp, 2);
            matches_cp_good = this->ratioTest(matches_cp);
        }
        catch(const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Feature matching failed: %s", e.what());
            return;
        }

        // Symmetry test
        this->symmetryTest(matches_pc_good, matches_cp_good, good_matches);
    }

    void feature_callback(const std::shared_ptr<vio_msgs::srv::FeatureExtractor::Request> request,std::shared_ptr<vio_msgs::srv::FeatureExtractor::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received feature extraction request. Size: " 
            << request->curr_img.width << "x" << request->curr_img.height);
        auto estimation_start = std::chrono::high_resolution_clock::now();
        
        // Convert data from request
        cv::Mat curr_img = OpenCVConversions::toCvImage(request->curr_img);
        cv::Mat prev_img_desc = OpenCVConversions::toCvImage(request->prev_img_desc);

        // Extract features from current image
        cv::Mat curr_img_desc;
        std::vector<cv::KeyPoint> curr_img_kp;
        this->featureExtract(curr_img, curr_img_kp, curr_img_desc);

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Feature extraction time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());
        estimation_start = std::chrono::high_resolution_clock::now();

        // Match features between current and previous image
        std::vector<cv::DMatch> good_matches;
        this->featureMatch(curr_img_desc, prev_img_desc, good_matches);

        estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Feature matching time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());

        // Convert to ros message
        response->curr_img_desc = OpenCVConversions::toRosImage(curr_img_desc);
        response->curr_img_kps = OpenCVConversions::toRosKeyPoints(curr_img_kp);
        response->good_matches = OpenCVConversions::toRosDMatches(good_matches);
    }

public:

    FeatureExtractorServer(): Node("feature_extractor_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting feature extractor server...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);

        YAML::Node main_config = YAML::LoadFile(config_file); 
        std::string preset_path = "/workspace/config/" + main_config["preset"].as<std::string>() + ".yaml";
        YAML::Node preset_config = YAML::LoadFile(preset_path);

        // Parse parameters
        std::string feature_extractor_service = preset_config["feature_extractor_service"].as<std::string>();
        this->distance_ratio = main_config["feature_matcher"]["distance_ratio"].as<double>();
        std::string extractor = main_config["feature_matcher"]["extractor"].as<std::string>();
        std::string matcher = main_config["feature_matcher"]["matcher"].as<std::string>();

        // Initialize service
        this->feature_server = this->create_service<vio_msgs::srv::FeatureExtractor>(feature_extractor_service, 
            std::bind(&FeatureExtractorServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        cv::Ptr<cv::flann::IndexParams> indexParams;
        cv::Ptr<cv::flann::SearchParams> searchParams;

        // Initialize feature extractor
        if ((extractor == "orb") || (extractor == "cuda_orb"))
        {
            if (extractor == "cuda_orb")
            {
                #ifdef USE_CUDA
                    this->cuda_orb_extractor = cv::cuda::ORB::create();
                #else
                    this->feat_extractor = cv::ORB::create();
                #endif
            }
            else
            {
                this->feat_extractor = cv::ORB::create();
            }

            if (matcher == "flann")
            {
                indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1); 
                searchParams = cv::makePtr<cv::flann::SearchParams>(50); 
            }
        }
        else if (extractor == "sift")
        {
            this->feat_extractor = cv::SIFT::create();

            if (matcher == "flann")
            {
                indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
                searchParams = cv::makePtr<cv::flann::SearchParams>(50);
            }
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Invalid feature extractor: %s", extractor.c_str());
        }

        if (matcher == "flann")
        {
            this->matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
        } else {
            this->matcher = cv::BFMatcher::create(cv::NORM_L2, false);
        }

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