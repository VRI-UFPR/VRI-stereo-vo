#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <jetson-utils/videoSource.h>
#include "image_converter.h"

#include "rclcpp/rclcpp.hpp"

#include "stereo_vo/msg/stereo_image.hpp"

class ImxPublisher : public rclcpp::Node
{
public:
    ImxPublisher(void) : Node("imx_publisher")
    {
        RCLCPP_INFO(this->get_logger(), "Starting camera publisher...");

        this->readConfig();
        this->initCaptures();
        
        // Used to convert to ROS image message
        this->image_cvt = new imageConverter();

        // Init timer and publisher
        this->publisher = this->create_publisher<stereo_vo::msg::StereoImage>(this->topic, 10);
        this->timer = this->create_wall_timer(std::chrono::milliseconds(1000 / 30), std::bind(&ImxPublisher::publish, this));

        RCLCPP_INFO(this->get_logger(), "Camera publisher has been started.");
    }

private:

    void readConfig(void)
    {
        // Parse parameters
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode cam_config = fs["sensors"]["cameras"]["devices"];
        this->cameras.resize(cam_config.size());

        RCLCPP_INFO(this->get_logger(), "Using %d cameras.", cam_config.size());

        fs["sensors"]["cameras"]["topic"] >> this->topic;

        for (size_t i = 0; i < cam_config.size(); i++)
        {
            cam_config[i]["input_stream"] >> this->cameras[i].input_stream;
            cam_config[i]["sample_rate"] >> this->cameras[i].sample_rate;
            cam_config[i]["resolution"] >> this->cameras[i].resolution;
        }
        fs.release();

        for (size_t i = 0; i < this->cameras.size(); i++)
        {
            RCLCPP_INFO(this->get_logger(), "Camera %d: %s", i, this->cameras[i].toString().c_str());
        }
    }

    void initCaptures(void)
    {
        // Initialize jetson utils camera capture
        for (size_t i = 0; i < this->cameras.size(); i++)
        {
            Camera* camera = &this->cameras[i];

            videoOptions video_options;
            video_options.width = camera->resolution[0];
            video_options.height = camera->resolution[1];
            video_options.frameRate = static_cast<float>(camera->sample_rate);
            // In our current setup, the camera is upside down.
            video_options.flipMethod = videoOptions::FlipMethod::FLIP_ROTATE_180;
            video_options.latency = 1;

            camera->cap = videoSource::Create(camera->input_stream.c_str(), video_options);

            if (camera->cap == nullptr)
            {
                RCLCPP_ERROR(this->get_logger(), "Could not open camera %d.", i);
            }
        }
    }

    void publish(void)
    {
        stereo_vo::msg::StereoImage msg;
        imageConverter::PixelType* nextFrame = NULL;

        for (size_t idx = 0; idx < this->cameras.size(); idx++)
        {
            Camera* camera = &this->cameras[idx];

            // Retrieve the next frame from the camera
            if(!camera->cap->Capture(&nextFrame, 1000))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to capture camera %d frame.", idx);
                return;
            }

            // Resize and convert to a ROS message
            if(!this->image_cvt->Resize(camera->cap->GetWidth(), camera->cap->GetHeight(), imageConverter::ROSOutputFormat))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to resize camera %d image converter.", idx);
                return;
            }

            if (idx == 0)
            {
                this->image_cvt->Convert(msg.left_image, imageConverter::ROSOutputFormat, nextFrame);
            } else {
                this->image_cvt->Convert(msg.right_image, imageConverter::ROSOutputFormat, nextFrame);
                msg.stereo = true;
            }
        }

        msg.header.stamp = this->now();
        this->publisher->publish(msg);

        RCLCPP_INFO_ONCE(this->get_logger(), "Publishing camera frames...");
    }
   
    struct Camera
    {
        std::string input_stream;
        int sample_rate;
        std::vector<int> resolution;
        videoSource *cap;

        Camera() : input_stream(""), sample_rate(0), resolution({0, 0}), cap(nullptr) {}
        std::string toString(void) 
        {
            return "Camera: " + input_stream + " " + std::to_string(sample_rate) + " " + std::to_string(resolution[0]) + "x" + std::to_string(resolution[1]);
        }
    };

    rclcpp::Publisher<stereo_vo::msg::StereoImage>::SharedPtr publisher;
    rclcpp::TimerBase::SharedPtr timer;
    std::string topic;

    std::vector<Camera> cameras;
    imageConverter* image_cvt;
};

int main(int argc, char *argv[])
{
    // initialize ros2
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImxPublisher>());
    rclcpp::shutdown();
    return 0;
}