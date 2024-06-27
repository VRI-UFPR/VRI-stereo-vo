#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cv_bridge/cv_bridge.h>

#include <jetson-utils/videoSource.h>
#include "image_converter.h"

#include "rclcpp/rclcpp.hpp"

#include "vio_msgs/msg/stereo_image.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImxPublisher : public rclcpp::Node
{
private:

    struct Camera
    {
        std::string input_stream;
        std::string topic;
        int sample_rate;
        std::vector<int> resolution;
        videoSource *cap;
        imageConverter* cvt;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher;
        rclcpp::TimerBase::SharedPtr timer;

        Camera() : input_stream(""), sample_rate(0), resolution({0, 0}), cap(nullptr), publisher(nullptr) {}
        std::string toString(void) 
        {
            return "Camera: " + input_stream + " " + std::to_string(sample_rate) + " " + std::to_string(resolution[0]) + "x" + std::to_string(resolution[1]);
        }
    };

    std::vector<Camera> cameras;

    void readConfig(void)
    {
        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config_imx.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);
        cv::FileStorage fs(config_file, cv::FileStorage::READ);

        // Parse parameters
        cv::FileNode cam_config = fs["sensors"]["cameras"]["devices"];
        this->cameras.resize(cam_config.size());

        RCLCPP_INFO(this->get_logger(), "Using %d cameras.", cam_config.size());

        for (size_t i = 0; i < cam_config.size(); i++)
        {
            cam_config[i]["input_stream"] >> this->cameras[i].input_stream;
            cam_config[i]["topic"] >> this->cameras[i].topic;
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
            camera->cvt = new imageConverter();

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
            Camera *camera = &this->cameras[i];

            camera->publisher = this->create_publisher<sensor_msgs::msg::Image>(camera->topic, 10);

            // Use std::bind to pass the camera to the callback function
            std::function<void()> callback = std::bind(&ImxPublisher::publish, this, camera);
            camera->timer = this->create_wall_timer(std::chrono::milliseconds(1000 / camera->sample_rate), callback);
        }
    }

    void publish(Camera *camera)
    {
        sensor_msgs::msg::Image msg;
        imageConverter::PixelType* nextFrame = NULL;

        // Retrieve the next frame from the camera
        if(!camera->cap->Capture(&nextFrame, 1000))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture camera %s frame.", camera->input_stream.c_str());
            return;
        }

        // Resize and convert to a ROS message
        if(!camera->cvt->Resize(camera->cap->GetWidth(), camera->cap->GetHeight(), imageConverter::ROSOutputFormat))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to resize camera %s image converter.", camera->input_stream.c_str());
            return;
        }

        camera->cvt->Convert(msg, imageConverter::ROSOutputFormat, nextFrame);

        msg.header.stamp = this->now();
        camera->publisher->publish(msg);

        RCLCPP_INFO_ONCE(this->get_logger(), "Publishing camera frames...");
    }

public:

    ImxPublisher(void) : Node("imx_publisher")
    {
        RCLCPP_INFO(this->get_logger(), "Starting camera publisher...");

        this->readConfig();
        this->initCaptures();
        this->initPublishers();

        RCLCPP_INFO(this->get_logger(), "Camera publisher has been started.");
    }
};

int main(int argc, char *argv[])
{
    // initialize ros2
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImxPublisher>());
    rclcpp::shutdown();
    return 0;
}