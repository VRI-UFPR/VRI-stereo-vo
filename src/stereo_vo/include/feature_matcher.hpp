#include <string>
#include <vector>
#include <chrono>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
    #include <opencv2/cudafeatures2d.hpp>
#endif

#include <opencv2/features2d.hpp>

class FeatureMatcher
{
private:

    #ifdef USE_CUDA
        cv::Ptr<cv::cuda::ORB> cuda_orb_extractor;
    #else
        cv::Ptr<cv::Feature2D> feat_extractor;
    #endif

    cv::Ptr<cv::DescriptorMatcher> matcher;

    double distance_ratio = 0.3;

    long total_feature_estimations = 0;
    long total_feature_estimation_time = 0;

    void featureExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
    std::vector<cv::DMatch> ratioTest(std::vector<std::vector<cv::DMatch>> &matches);
    void symmetryTest(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2, std::vector<cv::DMatch>& symMatches);
    void featureMatch(const cv::Mat &curr_img_desc, const cv::Mat &prev_img_desc, std::vector<cv::DMatch> &good_matches);

public:

    struct MatchResponse
    {
        cv::Mat curr_img_desc;
        std::vector<cv::KeyPoint> curr_img_kp;
        std::vector<cv::DMatch> good_matches;

        long extraction_time;
        long matching_time;
        long total_time;
        double average_time;
    };

    FeatureMatcher() {};
    FeatureMatcher(YAML::Node &config);
    std::shared_ptr<MatchResponse> compute(cv::Mat curr_img, cv::Mat prev_img_desc);
};


FeatureMatcher::FeatureMatcher(YAML::Node &config)
{
    // Parse parameters
    this->distance_ratio = config["distance_ratio"].as<double>();
    std::string extractor = config["extractor"].as<std::string>();
    std::string matcher = config["matcher"].as<std::string>();

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
    else
    {
        this->feat_extractor = cv::SIFT::create();

        if (matcher == "flann")
        {
            indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
            searchParams = cv::makePtr<cv::flann::SearchParams>(50);
        }
    }

    if (matcher == "flann")
    {
        this->matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
    } else {
        this->matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    }
}

std::shared_ptr<FeatureMatcher::MatchResponse> FeatureMatcher::compute(cv::Mat curr_img, cv::Mat prev_img_desc)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<MatchResponse> response = std::make_shared<MatchResponse>();

    // Extract features from current image
    auto estimation_start = std::chrono::high_resolution_clock::now();
    this->featureExtract(curr_img, response->curr_img_kp, response->curr_img_desc);
    auto estimation_end = std::chrono::high_resolution_clock::now();
    response->extraction_time = std::chrono::duration_cast<std::chrono::milliseconds>(estimation_end - estimation_start).count();

    // Match features between current and previous image
    estimation_start = std::chrono::high_resolution_clock::now();
    this->featureMatch(response->curr_img_desc, prev_img_desc, response->good_matches);
    estimation_end = std::chrono::high_resolution_clock::now();
    response->matching_time = std::chrono::duration_cast<std::chrono::milliseconds>(estimation_end - estimation_start).count();

    // Calculate total and average time
    auto total_end = std::chrono::high_resolution_clock::now();
    response->total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    this->total_feature_estimations++;
    this->total_feature_estimation_time += response->total_time;
    response->average_time = (double)this->total_feature_estimation_time / this->total_feature_estimations;

    return response;
}

void FeatureMatcher::featureExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
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

std::vector<cv::DMatch> FeatureMatcher::ratioTest(std::vector<std::vector<cv::DMatch>> &matches)
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

void FeatureMatcher::symmetryTest(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2, std::vector<cv::DMatch>& symMatches)
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

void FeatureMatcher::featureMatch(const cv::Mat &curr_img_desc, const cv::Mat &prev_img_desc, std::vector<cv::DMatch> &good_matches)
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
        return;
    }

    // Symmetry test
    this->symmetryTest(matches_pc_good, matches_cp_good, good_matches);
}