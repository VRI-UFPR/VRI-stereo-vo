#include <string>
#include <iostream>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
    #include <opencv2/cudastereo.hpp>
#endif

#include "opencv_conversions.hpp"

class DepthEstimator
{
private:
    
    cv::Ptr<cv::StereoMatcher> stereo_matcher = nullptr;
    bool using_cuda = false;
    bool initialized = false;
    cv::Size img_size{0, 0};
    OpenCVConversions::CameraIntrinsics lcam_intrinsics, rcam_intrinsics;

    void initStereoBM(const int ndisp, const int block_size);
    void initStereoSGBM(const int ndisp, const int block_size);
    void initCudaBM(const int ndisp, const int block_size);
    void initBeliefPropagation(const cv::Size im_size, const int ndisp, const int iter, const int levels);
    void initConstantSpaceBP(const cv::Size im_size, const int ndisp, const int iter, const int levels, const int nr_plane);

public:
    DepthEstimator() {};
    DepthEstimator(const YAML::Node &config, const std::string lcam_intrinsics_file, const std::string rcam_intrinsics_file, cv::Size img_size);

    cv::Mat compute(const cv::Mat &img_left, const cv::Mat &img_right, bool undistorted);
};

void DepthEstimator::initStereoBM(const int ndisp, const int block_size)
{
    int _ndisp = (ndisp > 0) ? ndisp : 64;
    int _block_size = (block_size > 0) ? block_size : 19;

    this->stereo_matcher = cv::StereoBM::create(_ndisp, _block_size);
    
    this->initialized = true;
}

void DepthEstimator::initStereoSGBM(const int ndisp, const int block_size)
{
    int _ndisp = (ndisp > 0) ? ndisp : 16;
    int _block_size = (block_size > 0) ? block_size : 3;

    this->stereo_matcher = cv::StereoSGBM::create(0, _ndisp, _block_size, 8 * _block_size * _block_size, 32 * _block_size * _block_size, 1, 63, 10, 100, 32, cv::StereoSGBM::MODE_SGBM_3WAY);

    this->initialized = true;
}

void DepthEstimator::initCudaBM(const int ndisp, const int block_size)
{
    #ifdef USE_CUDA
        int _ndisp = (ndisp > 0) ? ndisp : 64;
        int _block_size = (block_size > 0) ? block_size : 19;

        this->stereo_matcher = cv::cuda::createStereoBM(_ndisp, _block_size);

        this->initialized = true;
        this->using_cuda = true;
    #else
        // Avoid warning
        (void)ndisp; (void)block_size;
        std::cerr << "[Depth Estimator] CUDA not enabled. Cannot use cuda stereo matcher." << std::endl;
    #endif
}


void DepthEstimator::initBeliefPropagation(const cv::Size im_size, const int ndisp, const int iter, const int levels)
{
    #ifdef USE_CUDA
        // Estimate recommended parameters
        int r_ndisp, r_iter, r_levels;
        cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(im_size.width, im_size.height, r_ndisp, r_iter, r_levels);

        int _ndisp = (ndisp > 0) ? ndisp : r_ndisp;
        int _iter = (iter > 0) ? iter : r_iter;
        int _levels = (levels > 0) ? levels : r_levels;

        this->stereo_matcher = cv::cuda::createStereoBeliefPropagation(_ndisp, _iter, _levels, CV_16SC1);

        this->initialized = true;
        this->using_cuda = true;
    #else
        // Avoid warning
        (void)im_size; (void)ndisp; (void)iter; (void)levels;
        std::cerr << "[Depth Estimator] CUDA not enabled. Cannot use cuda stereo matcher." << std::endl;
    #endif
}

void DepthEstimator::initConstantSpaceBP(const cv::Size im_size, const int ndisp, const int iter, const int levels, const int nr_plane)
{
    #ifdef USE_CUDA
        // Estimate recommended parameters
        int r_ndisp, r_iter, r_levels, r_nr_plane;
        cv::cuda::StereoConstantSpaceBP::estimateRecommendedParams(im_size.width, im_size.height, r_ndisp, r_iter, r_levels, r_nr_plane);

        int _ndisp = (ndisp > 0) ? ndisp : r_ndisp;
        int _iter = (iter > 0) ? iter : r_iter;
        int _levels = (levels > 0) ? levels : r_levels;
        int _nr_plane = (nr_plane > 0) ? nr_plane : r_nr_plane;

        this->stereo_matcher = cv::cuda::createStereoConstantSpaceBP(_ndisp, _iter, _levels, _nr_plane);

        this->initialized = true;
        this->using_cuda = true;
    #else
        // Avoid warning
        (void)im_size; (void)ndisp; (void)iter; (void)levels; (void)nr_plane;
        std::cerr << "[Depth Estimator] CUDA not enabled. Cannot use cuda stereo matcher." << std::endl;
    #endif
}

DepthEstimator::DepthEstimator(const YAML::Node &config, const std::string lcam_intrinsics_file, const std::string rcam_intrinsics_file, cv::Size img_size)
{
    this->img_size = img_size;

    std::string algorithm = config["depth_algorithm"].as<std::string>();
    int ndisp = config["num_disparities"].as<int>();
    int block_size = config["block_size"].as<int>();
    int iter = config["iterations"].as<int>();
    int levels = config["levels"].as<int>();
    int nr_plane = config["nr_plane"].as<int>();

    this->lcam_intrinsics = OpenCVConversions::CameraIntrinsics(lcam_intrinsics_file);
    this->rcam_intrinsics = OpenCVConversions::CameraIntrinsics(rcam_intrinsics_file);

    std::cout << "[Depth Estimator] Using '" << algorithm << "'." << std::endl;

    if (algorithm == "stereo_bm")
        this->initStereoBM(ndisp, block_size);
    else if (algorithm == "stereo_sgbm")
        this->initStereoSGBM(ndisp, block_size);
    else if (algorithm == "cuda_bm")
        this->initCudaBM(ndisp, block_size);
    else if (algorithm == "belief_propagation")
        this->initBeliefPropagation(img_size, ndisp, iter, levels);
    else if (algorithm == "constant_space_bp")
        this->initConstantSpaceBP(img_size, ndisp, iter, levels, nr_plane);
}

cv::Mat DepthEstimator::compute(const cv::Mat &img_left, const cv::Mat &img_right, bool undistorted = false)
{
    if (!this->initialized)
    {
        std::cerr << "[Depth Estimator] Using stereo matcher not initialized." << std::endl;
        return cv::Mat(this->img_size, 0);
    }

    // Undistort images
    cv::Mat limg_rect = img_left;
    cv::Mat rimg_rect = img_right;
    if (!undistorted)
    {
        limg_rect = this->lcam_intrinsics.undistortImage(img_left);
        rimg_rect = this->rcam_intrinsics.undistortImage(img_right);
    }

    cv::Mat disparity_map;
    
    if (this->using_cuda)
    {
        #ifdef USE_CUDA
            cv::cuda::GpuMat img_left_gpu, img_right_gpu, disparity_map_gpu;

            // Upload images to GPU
            img_left_gpu.upload(limg_rect);
            img_right_gpu.upload(rimg_rect);

            // Compute disparity map
            stereo_matcher->compute(img_left_gpu, img_right_gpu, disparity_map_gpu);

            // Download depth map
            disparity_map_gpu.download(disparity_map);
        #endif
    } else {
        stereo_matcher->compute(limg_rect, rimg_rect, disparity_map);
    }

    // // Normalize
    cv::normalize(disparity_map, disparity_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // disparity_map.convertTo(disparity_map, CV_64F);

    // double fc = this->lcam_intrinsics.fx() * this->lcam_intrinsics.tlVector().at<double>(1);

    // cv::Mat depth_map = cv::Mat::zeros(disparity_map.size(), CV_64F);
    // cv::Mat mask = (disparity_map == 0.0);
    // disparity_map.setTo(0.001, mask);
    // disparity_map = fc / disparity_map;

    // cv::normalize(disparity_map, disparity_map, 0, 65536, cv::NORM_MINMAX, CV_16UC1);

    return disparity_map;
}
