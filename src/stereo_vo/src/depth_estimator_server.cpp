#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/srv/depth_estimator.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <cv_bridge/cv_bridge.h>

class DepthEstimatorServer : public rclcpp::Node
{
private:

    rclcpp::Service<stereo_vo::srv::DepthEstimator>::SharedPtr depth_server;

    static cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo_matcher;

    // Camera intrinsics
    struct CameraIntrinsics
    {
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
    } lcam_intrinsics, rcam_intrinsics; 

    void depth_callback(const std::shared_ptr<stereo_vo::srv::DepthEstimator::Request> request,
                        std::shared_ptr<stereo_vo::srv::DepthEstimator::Response> response)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), 
            "Received depth estimation request. Size: " 
            << request->stereo_image.left_image.width << "x" << request->stereo_image.left_image.height << ", " 
            << request->stereo_image.right_image.width << "x" << request->stereo_image.right_image.height);

        cv::Mat img_left = cv_bridge::toCvCopy(request->stereo_image.left_image, "mono8")->image;
        cv::Mat img_right = cv_bridge::toCvCopy(request->stereo_image.right_image, "mono8")->image;

        cv::Mat depth_map;
        this->stereo_matcher->compute(img_left, img_right, depth_map); 

        respose->depth_map = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", depth_map).toImageMsg();

        RCLCPP_INFO(this->get_logger(), "Depth estimation completed.");
    }

public:
    DepthEstimatorServer(): Node("depth_estimator_server")
    {
        RCLCPP_INFO(this->get_logger(), "Starting depth estimator server...");

        // Parse parameters
        std::string depth_estimator_service; 
        cv::Size img_size;
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode vo_config = fs["stereo_vo"];
        vo_config["depth_estimator_service"] >> depth_estimator_service;
        vo_config["image_size"] >> img_size;
        fs.release();

        // Initialize service
        this->depth_server = this->create_service<stereo_vo::srv::DepthEstimator>(depth_estimator_service, 
            std::bind(&DepthEstimatorServer::depth_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize stereo BM
        int ndisp, iters, levels;
        cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(img_size.width, img_size.height, ndisp, iters, levels);
        this->stereo_matcher = cv::cuda::createStereoBeliefPropagation(ndisp, iters, levels, CV_16SC1);        

        RCLCPP_INFO(this->get_logger(), "Depth estimator server started.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthEstimatorServer>());
    rclcpp::shutdown();
    return 0;
}
