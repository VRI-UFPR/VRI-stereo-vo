#include <string>
#include <iostream>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
    #include <opencv2/cudastereo.hpp>
#endif

class DepthEstimator
{
private:
    
    cv::Ptr<cv::StereoMatcher> stereo_matcher = nullptr;
    bool using_cuda = false;
    bool initialized = false;
    cv::Size img_size{0, 0};

    long total_estimations = 0;
    long total_estimation_time = 0;

    void initStereoBM(const int ndisp, const int block_size);
    void initStereoSGBM(const int ndisp, const int block_size, const int sad_window);
    void initCudaBM(const int ndisp, const int block_size);
    void initCudaSGM(const int ndisp);
    void initBeliefPropagation(const cv::Size im_size, const int ndisp, const int iter, const int levels);
    void initConstantSpaceBP(const cv::Size im_size, const int ndisp, const int iter, const int levels, const int nr_plane);

public:

    struct DepthResponse 
    {
        cv::Mat disparity_map;

        long estimation_time;
        double average_time;
    };

    DepthEstimator() {};
    DepthEstimator(const YAML::Node &config, cv::Size img_size);

    std::shared_ptr<DepthResponse> compute(const cv::Mat &img_left, const cv::Mat &img_right);
};

void DepthEstimator::initStereoBM(const int ndisp, const int block_size)
{
    int _ndisp = (ndisp > 0) ? ndisp : 64;
    int _block_size = (block_size > 0) ? block_size : 19;

    this->stereo_matcher = cv::StereoBM::create(_ndisp, _block_size);
    
    this->initialized = true;
}

void DepthEstimator::initStereoSGBM(const int ndisp, const int block_size, const int sad_window)
{
    int _ndisp = (ndisp > 0) ? ndisp : 16;
    int _block_size = (block_size > 0) ? block_size : 11;
    int _sad_window = (sad_window > 0) ? sad_window : 5;

    this->stereo_matcher = cv::StereoSGBM::create(0, _ndisp, _block_size, 8 * 3 * _sad_window * _sad_window, 32 * 3 * _sad_window * _sad_window, 1, 63, 10, 100, 32, cv::StereoSGBM::MODE_SGBM_3WAY);

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

void DepthEstimator::initCudaSGM(const int ndisp)
{
    #ifdef USE_CUDA
        // int _ndisp = (ndisp > 0) ? ndisp : 128;

        // this->stereo_matcher = cv::cuda::createStereoSGM(0, _ndisp);

        // this->initialized = true;
        // this->using_cuda = true;
        (void)ndisp;
        std::cerr << "[Depth Estimator] Cuda SGM is not implemented in this version of OpenCV." << std::endl;
    #else
        // Avoid warning
        (void)ndisp;
        std::cerr << "[Depth Estimator] CUDA not enabled. Cannot use cuda SGM." << std::endl;
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
        std::cerr << "[Depth Estimator] CUDA not enabled. Cannot use belief propagation." << std::endl;
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
        std::cerr << "[Depth Estimator] CUDA not enabled. Cannot use constant space BP." << std::endl;
    #endif
}

DepthEstimator::DepthEstimator(const YAML::Node &config, cv::Size img_size)
{
    this->img_size = img_size;

    std::string algorithm = config["depth_algorithm"].as<std::string>();
    int ndisp = config["num_disparities"].as<int>();
    int block_size = config["block_size"].as<int>();
    int iter = config["iterations"].as<int>();
    int levels = config["levels"].as<int>();
    int nr_plane = config["nr_plane"].as<int>();
    int sad_window = config["sad_window"].as<int>();

    std::cout << "[Depth Estimator] Using '" << algorithm << "'." << std::endl;

    if (algorithm == "stereo_bm")
        this->initStereoBM(ndisp, block_size);
    else if (algorithm == "stereo_sgbm")
        this->initStereoSGBM(ndisp, block_size, sad_window);
    else if (algorithm == "cuda_bm")
        this->initCudaBM(ndisp, block_size);
    else if (algorithm == "cuda_sgm")
        this->initCudaSGM(ndisp);
    else if (algorithm == "belief_propagation")
        this->initBeliefPropagation(img_size, ndisp, iter, levels);
    else if (algorithm == "constant_space_bp")
        this->initConstantSpaceBP(img_size, ndisp, iter, levels, nr_plane);
}

std::shared_ptr<DepthEstimator::DepthResponse> DepthEstimator::compute(const cv::Mat &img_left, const cv::Mat &img_right)
{
    auto estimation_start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<DepthResponse> response = std::make_shared<DepthResponse>();

    if (!this->initialized)
    {
        std::cerr << "[Depth Estimator] Using stereo matcher not initialized." << std::endl;
        response->estimation_time = 0;
        response->average_time = (double)this->total_estimation_time / this->total_estimations;
        response->disparity_map = cv::Mat();
        return response;
    }

    if (this->using_cuda)
    {
        #ifdef USE_CUDA
            cv::cuda::GpuMat img_left_gpu, img_right_gpu, disparity_map_gpu;

            // Upload images to GPU
            img_left_gpu.upload(img_left);
            img_right_gpu.upload(img_right);

            // Compute disparity map
            stereo_matcher->compute(img_left_gpu, img_right_gpu, disparity_map_gpu);

            // Download depth map
            disparity_map_gpu.download(disparity_map);
        #endif
    } else {
        stereo_matcher->compute(img_left, img_right, response->disparity_map);
    }

    // Normalize
    response->disparity_map.convertTo(response->disparity_map, CV_32F, 1.0/16.0);

    auto estimation_end = std::chrono::high_resolution_clock::now();
    response->estimation_time = std::chrono::duration_cast<std::chrono::milliseconds>(estimation_end - estimation_start).count();
    this->total_estimations++;
    this->total_estimation_time += response->estimation_time;
    response->average_time = (double)this->total_estimation_time / this->total_estimations;

    return response;
}
