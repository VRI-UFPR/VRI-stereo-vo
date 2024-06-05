#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>

class DepthEstimator
{
private:
    
    cv::Ptr<cv::StereoMatcher> stereo_matcher = nullptr;
    bool using_cuda = false;
    bool initialized = false;
    cv::Size img_size(0, 0);

    void initStereoBM(const int ndisp, const int block_size);
    void initStereoSGBM(const int ndisp, const int block_size);
    void initCudaBM(const int ndisp, const int block_size);
    void initBeliefPropagation(const cv::Size im_size, const int ndisp, const int iter, const int levels);
    void initConstantSpaceBP(const cv::Size im_size, const int ndisp, const int iter, const int levels, const int nr_plane);

public:
    DepthEstimator() {};
    DepthEstimator(const cv::FileNode &config);

    cv::Mat compute(const cv::Mat &img_left, const cv::Mat &img_right);
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

    this->stereo_matcher = cv::StereoSGBM::create(0, _ndisp, _block_size);

    this->initialized = true;
}

void DepthEstimator::initCudaBM(const int ndisp, const int block_size)
{
    int _ndisp = (ndisp > 0) ? ndisp : 64;
    int _block_size = (block_size > 0) ? block_size : 19;

    this->stereo_matcher = cv::cuda::createStereoBM(_ndisp, _block_size);

    this->initialized = true;
    this->using_cuda = true;
}


void DepthEstimator::initBeliefPropagation(const cv::Size im_size, const int ndisp, const int iter, const int levels)
{
    // Estimate recommended parameters
    int r_ndisp, r_iter, r_levels;
    cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(im_size.width, im_size.height, r_ndisp, r_iter, r_levels);

    int _ndisp = (ndisp > 0) ? ndisp : r_ndisp;
    int _iter = (iter > 0) ? iter : r_iter;
    int _levels = (levels > 0) ? levels : r_levels;

    this->stereo_matcher = cv::cuda::createStereoBeliefPropagation(_ndisp, _iter, _levels, CV_16SC1);

    this->initialized = true;
    this->using_cuda = true;
}

void DepthEstimator::initConstantSpaceBP(const cv::Size im_size, const int ndisp, const int iter, const int levels, const int nr_plane)
{
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
}

DepthEstimator::DepthEstimator(const cv::FileNode &config)
{
    std::string algorithm = config["depth_algorithm"];

    cv::Size img_size = cv::Size(config["image_size"]["width"], config["image_size"]["height"]);
    this->img_size = img_size;
    int ndisp = config["num_disparities"];
    int block_size = config["block_size"];
    int iter = config["iterations"];
    int levels = config["levels"];
    int nr_plane = config["nr_plane"];

    switch (algorithm)
    {
    case "stereo_bm":
        this->initStereoBM(ndisp, block_size);
        break;

    case "stereo_sgbm":
        this->initStereoSGBM(ndisp, block_size);
        break;

    case "cuda_bm":
        this->initCudaBM(ndisp, block_size);
        break;

    case "belief_propagation":
        this->initBeliefPropagation(img_size, ndisp, iter, levels);
        break;

    case "constant_space_bp":
        this->initConstantSpaceBP(img_size, ndisp, iter, levels, nr_plane);
        break;
    
    default:
        break;
    }
}

cv::Mat DepthEstimator::compute(const cv::Mat &img_left, const cv::Mat &img_right)
{
    if (!this->initialized)
    {
        return cv::Mat(this->img_size, 0);
    }

    cv::Mat depth_map;
    
    if (this->using_cuda)
    {
        cv::cuda::GpuMat img_left_gpu, img_right_gpu, depth_map_gpu;

        // Upload images to GPU
        img_left_gpu.upload(img_left);
        img_right_gpu.upload(img_right);

        // Compute disparity map
        stereo_matcher->compute(img_left_gpu, img_right_gpu, depth_map_gpu);

        // Download depth map
        depth_map_gpu.download(depth_map);
    } else {
        stereo_matcher->compute(img_left, img_right, depth_map);
    }

    return depth_map;
}
