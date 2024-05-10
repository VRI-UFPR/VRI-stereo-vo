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
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImxPublisher : public rclcpp::Node
{
public:
    ImxPublisher(void) : Node("imx_publisher")
    {
        RCLCPP_INFO(this->get_logger(), "Starting camera publisher...");

        // parse parameters
        std::string camera_topic, input_stream;
        int sample_rate;
        std::vector<int> resolution;
        cv::FileStorage fs;
        fs.open("/workspace/config/config.yaml", cv::FileStorage::READ);
        fs["sensors"]["camera"]["topic"] >> camera_topic;
        fs["sensors"]["camera"]["sample_rate"] >> sample_rate;
        fs["sensors"]["camera"]["input_stream"] >> input_stream;
        fs["sensors"]["camera"]["resolution"] >> resolution;
        fs.release();

        videoOptions video_options;
        video_options.width = resolution[0];
        video_options.height = resolution[1];
        video_options.frameRate = static_cast<float>(sample_rate);
        // In our current setup, the camera is upside down.
        video_options.flipMethod = videoOptions::FlipMethod::FLIP_ROTATE_180;
        video_options.latency = 1;

        this->cap = videoSource::Create(input_stream.c_str(), video_options);

        if (this->cap == nullptr)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not open camera.");
        }

        this->image_cvt = new imageConverter();

        RCLCPP_INFO(this->get_logger(), "Camera publisher has been started.");

        // initialize publisher
        this->publisher = this->create_publisher<sensor_msgs::msg::Image>(camera_topic, 10);
        this->timer = this->create_wall_timer(std::chrono::milliseconds(1000 / sample_rate), std::bind(&ImxPublisher::publish, this));
    }

private:

    void publish(void)
    {
        sensor_msgs::msg::Image msg;
        imageConverter::PixelType* nextFrame = NULL;

        if(!this->cap->Capture(&nextFrame, 1000))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture camera frame.");
            return;
        }



        if(!this->image_cvt->Resize(this->cap->GetWidth(), this->cap->GetHeight(), imageConverter::ROSOutputFormat))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to resize camera image converter.");
            return;
        }

        this->image_cvt->Convert(msg, imageConverter::ROSOutputFormat, nextFrame);


        RCLCPP_INFO_ONCE(this->get_logger(), "Publishing camera frames...");

        msg.header.stamp = this->now();
        publisher->publish(msg);
    }

    videoSource* cap; 
    imageConverter* image_cvt;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher;
    rclcpp::TimerBase::SharedPtr timer;
};

int main(int argc, char *argv[])
{
    // initialize ros2
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImxPublisher>());
    rclcpp::shutdown();
    return 0;
}