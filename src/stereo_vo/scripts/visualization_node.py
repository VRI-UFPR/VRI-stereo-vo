#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import yaml

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tf2_msgs.msg import TFMessage

COLORS = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
]

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.get_logger().info("Starting visualization node...")

        # Load config
        config_file = self.declare_parameter("config_file", "/workspace/config/config.yaml").get_parameter_value().string_value
        self.get_logger().info(f"Loading config file: {config_file}")
        
        with open(config_file, 'r') as file:
            main_config = yaml.safe_load(file)

        preset_path = f"/workspace/config/{main_config['preset']}.yaml"
        with open(preset_path, 'r') as file:
            preset_config = yaml.safe_load(file)

        # Parse parameters
        visualization_topic = preset_config['vo_odom_topic']
        ground_truth = preset_config['ground_truth']
        visualization_topics = [visualization_topic, ground_truth]

        # Initialize messages static parameters
        self.msgs = []
        for i, topic in enumerate(visualization_topics):
            if not topic:
                continue

            msg = Marker()
            msg.header.frame_id = 'world'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.ns = 'points_and_lines'
            msg.action = Marker.ADD
            msg.id = i
            msg.type = Marker.LINE_STRIP
            msg.pose.orientation.w = 1.0
            msg.scale.x = 0.3
            msg.scale.y = 0.3
            msg.scale.z = 0.3

            msg.color.r = COLORS[i][0]
            msg.color.g = COLORS[i][1]
            msg.color.b = COLORS[i][2]
            msg.color.a = 1.0

            self.msgs.append(msg)

        # Initialize subscribers
        self.pose_subscribers = []
        self.markers_publishers = []

        callback = lambda msg: self.pose_callback(msg, 0)
        self.pose_subscribers.append(self.create_subscription(Odometry, visualization_topics[0], callback, 10))
        self.get_logger().info(f"/visualization{visualization_topics[0]}")
        self.markers_publishers.append(self.create_publisher(Marker, f"/visualization{visualization_topics[0]}", 10))

        if ground_truth:
            callback = lambda msg: self.transform_callback(msg, 1)
            self.create_subscription(TFMessage, visualization_topics[1], callback, 10)
            self.get_logger().info(f"/visualization{visualization_topics[1]}")
            self.markers_publishers.append(self.create_publisher(Marker, f"/visualization{visualization_topics[1]}", 10))

        self.get_logger().info("Visualization node has been started.")

    def pose_callback(self, msg, idx):
        self.get_logger().info("Started publishing markers...", once=True)
        self.msgs[idx].header.stamp = self.get_clock().now().to_msg()
        self.msgs[idx].header.frame_id = "camera"
        self.msgs[idx].points.append(msg.pose.pose.position)
        self.markers_publishers[idx].publish(self.msgs[idx])

    def point_callback(self, msg, idx):
        self.get_logger().info("Started publishing markers...", once=True)
        self.msgs[idx].header.stamp = self.get_clock().now().to_msg()
        self.msgs[idx].header.frame_id = "world"
        self.msgs[idx].points.append(msg.point)
        self.markers_publishers[idx].publish(self.msgs[idx])

    def transform_callback(self, msg, idx):
        self.get_logger().info("Started publishing markers...", once=True)
        self.msgs[idx].header.stamp = self.get_clock().now().to_msg()
        self.msgs[idx].header.frame_id = "camera"
        tf_point = PointStamped()
        tf_point.point.x = msg.transforms[0].transform.translation.x
        tf_point.point.y = msg.transforms[0].transform.translation.y
        tf_point.point.z = msg.transforms[0].transform.translation.z
        self.msgs[idx].points.append(tf_point.point)
        self.markers_publishers[idx].publish(self.msgs[idx])

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
