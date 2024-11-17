#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from os import makedirs
from os.path import exists

from std_msgs.msg import Float32

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image

COLORS = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
]

SCALE = 0.3

class VisualizerNode(Node):

    def __init__(self):
        super().__init__('visualization_node')
        self.get_logger().info("Starting visualization node...")

        MESSAGE_TYPES = {
            'odometry' : (Odometry, self.estm_callback),
            'point' : (PointStamped, self.point_callback),
            'transform' : (TFMessage, self.transform_callback),
        }

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
        ground_truth_msg_type = preset_config['ground_truth_msg_type']
        depth_topic = preset_config['depth_viz']
        self.gt_max_pts = main_config['stereo_vo']['buffer_size']

        # Initialize messages static parameters
        self.msgs = []
        for i, topic in enumerate([visualization_topic, ground_truth]):
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
            msg.scale.x = SCALE
            msg.scale.y = SCALE
            msg.scale.z = SCALE

            msg.color.r = COLORS[i][0]
            msg.color.g = COLORS[i][1]
            msg.color.b = COLORS[i][2]
            msg.color.a = 1.0

            self.msgs.append(msg)

        callback = lambda msg: self.estm_callback(msg, 0)
        self.estimation_subscriber = self.create_subscription(Odometry, visualization_topic, callback, 10)
        self.depth_subscriber = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.color_depth_pub = self.create_publisher(Image, "/visualization/depth_color", 10)
        self.estimation_publisher = self.create_publisher(Marker, f"/visualization{visualization_topic}", 10)
        self.get_logger().info(f"/visualization{visualization_topic}")

        if ground_truth:
            msg_type, gt_callback = MESSAGE_TYPES[ground_truth_msg_type]

            callback = lambda msg: gt_callback(msg, 1)
            self.gt_subscriber = self.create_subscription(msg_type, ground_truth, callback, 10)
            self.gt_publisher = self.create_publisher(Marker, f"/visualization{ground_truth}", 10)
            self.get_logger().info(f"/visualization{ground_truth}")

        # Error metrics
        self.last_estm = None
        self.last_gt = []
        self.total_he_error = 0.0
        self.total_tde_error = 0.0
        self.msg_num = 0

        self.date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Error publishers
        self.horizontal_error_pub = self.create_publisher(Float32, "/visualization/horizontal_error", 10)
        self.vertical_error_pub = self.create_publisher(Float32, "/visualization/vertical_error", 10)
        self.mean_tde_pub = self.create_publisher(Float32, "/visualization/mean_3dof_error", 10)
        
        # Plot points
        self.horizontal_error_pts = []
        self.vertical_error_pts = []
        self.mean_tde_error_pts = []
        self.path_pts = []
        self.gt_pts = []

        self.cv_bridge = CvBridge()

        self.get_logger().info("Visualization node has been started.")

    def depth_callback(self, msg):
        depth_image = self.cv_bridge.imgmsg_to_cv2(msg)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(depth_image)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        img_size = (depth_image.shape[0] - 75, depth_image.shape[1])
        crop_start = (image.shape[0] - img_size[0]) // 2
        crop_end = crop_start + img_size[0]
        image = image[crop_start:crop_end, :, :]

        color_depth_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="rgba8")
        color_depth_msg.header = msg.header
        self.color_depth_pub.publish(color_depth_msg)

    def estm_callback(self, msg, idx):
        self.get_logger().info("Started publishing markers...", once=True)
        self.msgs[idx].header.stamp = self.get_clock().now().to_msg()
        self.msgs[idx].header.frame_id = "camera"
        self.msgs[idx].points.append(msg.pose.pose.position)
        self.estimation_publisher.publish(self.msgs[idx])

        self.last_estm = np.array([msg.pose.pose.position.z, msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.msg_num += 1

        self.path_pts.append(self.last_estm[:2])

        self.publish_errors()

    def point_callback(self, msg, idx):
        self.get_logger().info("Started publishing markers...", once=True)
        self.msgs[idx].header.stamp = self.get_clock().now().to_msg()
        self.msgs[idx].header.frame_id = "world"
        self.msgs[idx].points.append(msg.point)
        self.gt_publisher.publish(self.msgs[idx])
        self.last_gt.append(np.array([msg.point.x, msg.point.y, msg.point.z]))
        self.gt_pts.append(self.last_gt[-1][:2])

        if len(self.last_gt) > self.gt_max_pts:
            self.last_gt.pop(0)

    def transform_callback(self, msg, idx):
        self.get_logger().info("Started publishing markers...", once=True)
        self.msgs[idx].header.stamp = self.get_clock().now().to_msg()
        self.msgs[idx].header.frame_id = "camera"
        tf_point = PointStamped()
        tf_point.point.x = msg.transforms[0].transform.translation.x
        tf_point.point.y = msg.transforms[0].transform.translation.y
        tf_point.point.z = msg.transforms[0].transform.translation.z
        self.msgs[idx].points.append(tf_point.point)
        self.gt_publisher.publish(self.msgs[idx])
        self.last_gt.append(np.array([tf_point.point.z, tf_point.point.x, tf_point.point.y]))
        self.gt_pts.append(self.last_gt[-1][:2])

        if len(self.last_gt) > self.gt_max_pts:
            self.last_gt.pop(0)
        
    def publish_errors(self):
        if len(self.last_gt) > 0:
            curr_gt = self.last_gt.pop(0)

            msg = Float32()
            
            # Publish instantaneous horizontal error
            horizontal_error = abs(np.linalg.norm(curr_gt[:2] - self.last_estm[:2])) 
            msg.data = horizontal_error
            self.horizontal_error_pts.append(horizontal_error)
            self.horizontal_error_pub.publish(msg)

            # Publish intaneous vertical error
            vertical_error = abs(curr_gt[2] - self.last_estm[2])
            msg.data = vertical_error
            self.vertical_error_pts.append(vertical_error)
            self.vertical_error_pub.publish(msg)

            # Publish mean 3DoF error
            self.total_tde_error += np.linalg.norm(curr_gt - self.last_estm)
            mean_tde_error = self.total_tde_error / self.msg_num
            msg.data = mean_tde_error
            self.mean_tde_error_pts.append(mean_tde_error)
            self.mean_tde_pub.publish(msg)

            if self.msg_num % 30 == 0:
                self.plot_erros()

    def plot_erros(self):

        if not exists(f"/workspace/Data/plots/{self.date_str}"):
            makedirs(f"/workspace/Data/plots/{self.date_str}")
    
        # Plot instantaneous errors
        plt.figure()
        plt.title("Instantaneous Errors (m)")
        plt.grid()
        plt.plot(self.horizontal_error_pts, label="Horizontal Error")
        plt.plot(self.vertical_error_pts, label="Vertical Error")
        plt.plot(self.mean_tde_error_pts, label="Mean 3DoF Error")
        plt.xlabel("Points")
        plt.ylabel("Error (m)")

        plt.legend()
        plt.savefig(f"/workspace/Data/plots/{self.date_str}/errors.png")

        # Plot paths
        plt.figure()
        plt.title("Paths (m)")
        plt.grid()
        path_pts = np.array(self.path_pts)
        gt_pts = np.array(self.gt_pts)
        plt.plot(path_pts[:, 0], path_pts[:, 1], label="Estimated Path")
        plt.plot(gt_pts[:, 0], gt_pts[:, 1], label="Ground Truth Path")
        plt.legend()
        plt.savefig(f"/workspace/Data/plots/{self.date_str}/paths.png")

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
