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

        this->readConfig();
        this->initCaptures();
        this->initPublishers();
        
        // Used to convert to ROS image message
        this->image_cvt = new imageConverter();

        RCLCPP_INFO(this->get_logger(), "Camera publisher has been started.");
    }

private:

    void readConfig(void)
    {
        // Parse parameters
        cv::FileStorage fs("/workspace/config/config.yaml", cv::FileStorage::READ);
        cv::FileNode cam_config = fs["sensors"]["cameras"];
        this->cameras.resize(cam_config.size());

        for (size_t i = 0; i < cam_config.size(); i++)
        {
            cam_config[i]["topic"] >> this->cameras[i].topic;
            cam_config[i]["input_stream"] >> this->cameras[i].input_stream;
            cam_config[i]["sample_rate"] >> this->cameras[i].sample_rate;
            cam_config[i]["resolution"] >> this->cameras[i].resolution;
        }
        fs.release();
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

    void initPublishers(void)
    {
        for (size_t i = 0; i < this->cameras.size(); i++)
        {
            Camera* camera = &this->cameras[i];

            camera->publisher = this->create_publisher<sensor_msgs::msg::Image>(camera->topic, 10);

            // Use std::bind to pass the camera index to the callback function
            std::function<void()> callback = std::bind(&ImxPublisher::publish, this, i);
            camera->timer = this->create_wall_timer(std::chrono::milliseconds(1000 / camera->sample_rate), callback);
        }
    }

    void publish(size_t idx)
    {
        sensor_msgs::msg::Image msg;
        imageConverter::PixelType* nextFrame = NULL;

        Camera* camera = &this->cameras[idx];

        // Retrieve the next frame from the camera
        if(!camera->cap->Capture(&nextFrame, 1000))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture camera frame.");
            return;
        }

        // Resize and convert to a ROS message
        if(!this->image_cvt->Resize(camera->cap->GetWidth(), camera->cap->GetHeight(), imageConverter::ROSOutputFormat))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to resize camera image converter.");
            return;
        }
        this->image_cvt->Convert(msg, imageConverter::ROSOutputFormat, nextFrame);

        msg.header.stamp = this->now();
        camera->publisher->publish(msg);

        RCLCPP_INFO_ONCE(this->get_logger(), "Publishing camera frames...");
    }
   
    struct Camera
    {
        std::string topic;
        std::string input_stream;
        int sample_rate;
        std::vector<int> resolution;
        videoSource *cap;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher;
        rclcpp::TimerBase::SharedPtr timer;

        Camera() : topic(""), input_stream(""), sample_rate(0), resolution({0, 0}), cap(nullptr) {}
    };

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