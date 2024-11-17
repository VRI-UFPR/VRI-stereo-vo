from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, AndSubstitution, NotSubstitution
from launch_ros.actions import Node

import yaml

BAG_FOLDER = "/workspace/Data/bags"
CONFIGS_FOLDER = "/workspace/config"
MAIN_CONFIG_PATH = CONFIGS_FOLDER + "/config.yaml"

def bool2str(value):
    return "true" if value else "false"

def parse_config_file():
    
    # Load main config file
    with open(MAIN_CONFIG_PATH, 'r') as stream:
        main_config = yaml.safe_load(stream)

    # Load preset config file
    with open(f"{CONFIGS_FOLDER}/{main_config['preset']}.yaml", 'r') as stream:
        preset_config = yaml.safe_load(stream)
        
    launch_args = [

        # Sensors arguments
        DeclareLaunchArgument(name="camera_enable", default_value=bool2str(preset_config["camera"]), description="Enable camera node"),
        DeclareLaunchArgument(name="imu_enable", default_value=bool2str(preset_config["imu"]), description="Enable IMU node"),

        # VO arguments
        DeclareLaunchArgument(name="vo_enable", default_value=bool2str(preset_config["vo"]), description="Enable VO pose estimation"),
        DeclareLaunchArgument(name="estimate_depth", default_value="true", description="Run depth estimator server"),
        DeclareLaunchArgument(name="enable_viz", default_value=bool2str(preset_config["markers"]), description="Enable markers visualization"),

        # Bag arguments
        DeclareLaunchArgument(name="bag", default_value=bool2str(preset_config["bag"]), description="Play bag file"),
        DeclareLaunchArgument(name="bagfile", default_value=preset_config["bag_name"], description="Bag file name"),
        DeclareLaunchArgument(name="pb_rate", default_value=str(preset_config["bag_rate"]), description="Playback rate"),
    ]

    return launch_args

def launch_setup(context):
    bag_path = PathJoinSubstitution([BAG_FOLDER, LaunchConfiguration("bagfile")])

    return [
        # Launch drivers
        Node(
            package='drivers',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("camera_enable"), NotSubstitution(LaunchConfiguration("bag")))),
            executable='camera_publisher',
            parameters=[{'config_file' : MAIN_CONFIG_PATH}],
        ),
        Node(
            package='drivers',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("imu_enable"), NotSubstitution(LaunchConfiguration("bag")))),
            executable='bno_publisher.py',
            parameters=[{'config_file' : MAIN_CONFIG_PATH}],
        ),

        # Launch VO related nodes
        Node(   
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("vo_enable")),
            executable='stereo_vo_node',
            parameters=[{'config_file' : MAIN_CONFIG_PATH}],
        ),

        # Launch visualization
        Node(
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("enable_viz")),
            executable='visualization_node.py',
            parameters=[{'config_file' : MAIN_CONFIG_PATH}],
        ),
        Node(
            package='tf2_ros',
            condition=IfCondition(LaunchConfiguration("enable_viz")),
            executable='static_transform_publisher',
            arguments=['--roll', '1.57079633', '--pitch', '1.57079633', '--yaw', '0.0', '--frame-id', 'camera', '--child-frame-id', 'world'],
        ),

        # Launch bag file
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("bag")),
            cmd=["ros2", "bag", "play", "-r", LaunchConfiguration("pb_rate"), bag_path],
            output="screen"
        )
    ]

def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    launch_args = parse_config_file()
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld