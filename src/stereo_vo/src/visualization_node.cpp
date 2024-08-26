#include "rclcpp/rclcpp.hpp"

#include <vector>
#include <string>

#include <yaml-cpp/yaml.h>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <visualization_msgs/msg/marker.hpp>

const std::vector<std::vector<float>> COLORS = {
    {0.0, 1.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
};

class VisualizerNode : public rclcpp::Node
{
public:
    VisualizerNode (void) : Node("visualization_node")
    {
        RCLCPP_INFO(this->get_logger(), "Starting visualization node...");

        // Load config
        std::string config_file;
        this->declare_parameter("config_file", "/workspace/config/config.yaml");
        this->get_parameter("config_file", config_file);
        RCLCPP_INFO_STREAM(this->get_logger(), "Loading config file: " << config_file);
        
        YAML::Node main_config = YAML::LoadFile(config_file); 
        std::string preset_path = "/workspace/config/" + main_config["preset"].as<std::string>() + ".yaml";
        YAML::Node preset_config = YAML::LoadFile(preset_path);

        // Parse parameters
        std::string visualization_topic = preset_config["vo_odom_topic"].as<std::string>();
        std::string ground_truth = preset_config["ground_truth"].as<std::string>();
        
        std::vector<std::string> visualization_topics = {visualization_topic, ground_truth};

        // Intialize messages static parameters
        for (size_t i = 0; i < visualization_topics.size(); i++)
        {
            if (visualization_topics[i].empty())
                continue;

            visualization_msgs::msg::Marker msg;
            msg.header.frame_id = "world";
            msg.header.stamp = this->now();
            msg.ns = "points_and_lines";
            msg.action = visualization_msgs::msg::Marker::ADD;
            msg.id = i;
            msg.type = visualization_msgs::msg::Marker::LINE_STRIP; 
            msg.pose.orientation.w = 1.0;

            msg.scale.x = 0.1; 
            msg.scale.y = 0.1;
            msg.scale.z = 0.1;

            msg.color.r = COLORS[i][0];
            msg.color.g = COLORS[i][1];
            msg.color.b = COLORS[i][2];
            msg.color.a = 1.0;

            this->msgs.push_back(msg);
        }

        // Initialize subscribers
        std::function<void(const nav_msgs::msg::Odometry::SharedPtr msg)> callback = std::bind(&VisualizerNode::pose_callback, this, std::placeholders::_1, 0);
        this->pose_subscribers.push_back(this->create_subscription<nav_msgs::msg::Odometry>(visualization_topics[0], 10, callback));

        RCLCPP_INFO(this->get_logger(), ("/visualization" + visualization_topics[0]).c_str());
        this->markers_publishers.push_back(this->create_publisher<visualization_msgs::msg::Marker>("/visualization" + visualization_topics[0], 10));

        if (!ground_truth.empty())
        {
            std::function<void(const geometry_msgs::msg::PointStamped::SharedPtr msg)> point_callback = std::bind(&VisualizerNode::point_callback, this, std::placeholders::_1, 1);
            this->point_subscribers.push_back(this->create_subscription<geometry_msgs::msg::PointStamped>(visualization_topics[1], 10, point_callback));

            RCLCPP_INFO(this->get_logger(), ("/visualization" + visualization_topics[1]).c_str());
            this->markers_publishers.push_back(this->create_publisher<visualization_msgs::msg::Marker>("/visualization" + visualization_topics[1], 10));
        }


        RCLCPP_INFO(this->get_logger(), "Visuzalition node has been started.");
    }

private:
    
    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr msg, int idx)
    {
        RCLCPP_INFO_ONCE(this->get_logger(), "Started publishing markers...");

        this->msgs[idx].header.stamp = this->now();
        this->msgs[idx].points.push_back(msg->pose.pose.position);
        this->markers_publishers[idx]->publish(this->msgs[idx]);
    }

    void point_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg, int idx)
    {
        RCLCPP_INFO_ONCE(this->get_logger(), "Started publishing markers...");

        this->msgs[idx].header.stamp = this->now();
        this->msgs[idx].points.push_back(msg->point);
        this->markers_publishers[idx]->publish(this->msgs[idx]);
    }

    std::vector<visualization_msgs::msg::Marker> msgs;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> pose_subscribers;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr> point_subscribers;
    std::vector<rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr> markers_publishers;
};

int main(int argc, char const *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualizerNode>());
    rclcpp::shutdown();

    return 0;
}