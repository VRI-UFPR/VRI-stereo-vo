#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/feature_extractor.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "opencv_conversions.hpp"

#include <string>
#include <vector>
#include <chrono>

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
        cv::Ptr<cv::cuda::ORB> orb_detector;
    #else
        cv::Ptr<cv::ORB> orb_detector;
    #endif

    cv::Ptr<cv::DescriptorMatcher> matcher;

    void featureExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
    {
        #ifdef USE_CUDA
            // Upload to gpu
            cv::cuda::GpuMat cuda_img;
            cuda_img.upload(img);

            cv::cuda::GpuMat cuda_keypoints, cuda_descriptors;
            this->orb_detector->detectAndComputeAsync(cuda_img, cv::cuda::GpuMat(), cuda_keypoints, cuda_descriptors);

            // Download from gpu
            this->orb_detector->convert(cuda_keypoints, keypoints);
            cuda_descriptors.download(descriptors);
        #else
            this->orb_detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        #endif
    }

    void ratioTest(std::vector<std::vector<cv::DMatch> > &matches)
    {
        // As per Lowe's ratio test
        for ( std::vector<std::vector<cv::DMatch> >::iterator matchIterator= matches.begin(); matchIterator!= matches.end(); ++matchIterator)
        {
            if (matchIterator->size() > 1)
            {
                if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > 0.7)
                {
                    matchIterator->clear(); 
                }
            }
            else
            { 
                matchIterator->clear(); 
            }
        }
    }

    void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1, const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches)
    {
        for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
        {
            if (matchIterator1->empty() || matchIterator1->size() < 2)
                continue;

            for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
            {
                if (matchIterator2->empty() || matchIterator2->size() < 2)
                    continue;

                // Match symmetry test
                if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
                {
                    symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
                    break;
                }
            }
        }
    }

    void featureMatch(const cv::Mat &curr_img_desc, const cv::Mat &prev_img_desc, std::vector<cv::DMatch> &good_matches)
    {
        std::vector<std::vector<cv::DMatch>> matches_pc, matches_cp;

        // Match features
        if (curr_img_desc.empty() || prev_img_desc.empty())
        {
            return;
        }

        this->matcher->knnMatch(prev_img_desc, curr_img_desc, matches_pc, 2);
        this->matcher->knnMatch(curr_img_desc, prev_img_desc, matches_cp, 2);

        // Remove matches per Lewe's ratio test
        this->ratioTest(matches_pc);
        this->ratioTest(matches_cp);

        this->symmetryTest(matches_pc, matches_cp, good_matches);    
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

        // Match features between current and previous image
        std::vector<cv::DMatch> good_matches;
        this->featureMatch(curr_img_desc, prev_img_desc, good_matches);

        // Convert to ros message
        response->curr_img_desc = OpenCVConversions::toRosImage(curr_img_desc);
        response->curr_img_kps = OpenCVConversions::toRosKeyPoints(curr_img_kp);
        response->good_matches = OpenCVConversions::toRosDMatches(good_matches);

        auto estimation_end = std::chrono::high_resolution_clock::now();
        RCLCPP_DEBUG(this->get_logger(), "Feature extraction time: %f ms", 
            std::chrono::duration<double, std::milli>(estimation_end - estimation_start).count());
        RCLCPP_INFO(this->get_logger(), "Feature extraction completed.");
    }

public:

    FeatureExtractorServer(): Node("feature_extractor_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting feature extractor server...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config_imx.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);
        cv::FileStorage fs(config_file, cv::FileStorage::READ);

        // Parse parameters
        cv::FileNode vo_config = fs["stereo_vo"];
        std::string feature_extractor_service = vo_config["feature_extractor_service"].string();
        fs.release();

        // Initialize service
        this->feature_server = this->create_service<vio_msgs::srv::FeatureExtractor>(feature_extractor_service, 
            std::bind(&FeatureExtractorServer::feature_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize feature extractor
        #ifdef USE_CUDA
            this->orb_detector = cv::cuda::ORB::create();
        #else
            this->orb_detector = cv::ORB::create();
        #endif

        // instantiate LSH index parameters
        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1); 
        // instantiate flann search parameters
        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50); 

        this->matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
        // this->matcher->train();

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