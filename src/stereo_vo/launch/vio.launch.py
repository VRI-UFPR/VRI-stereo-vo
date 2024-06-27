from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, AndSubstitution, NotSubstitution
from launch_ros.actions import Node

BAG_FOLDER = "/workspace/Data/bags/"
CONFIGS_FOLDER = "/workspace/config/"

launch_args = [
    DeclareLaunchArgument(name="camera_enable", default_value="true", description="Enable camera node"),
    DeclareLaunchArgument(name="imu_enable", default_value="true", description="Enable IMU node"),
    DeclareLaunchArgument(name="vo_enable", default_value="true", description="Enable VO pose estimation"),
    DeclareLaunchArgument(name="estimate_depth", default_value="false", description="Run depth estimator server"),

    # Bag arguments
    DeclareLaunchArgument(name="bag", default_value="true", description="Enable bag file"),
    DeclareLaunchArgument(name="bagfile", default_value="dataset_vri4wd_ufpr-map_20230830s001ab_ros2-bag", description="Bag folder to play"),

    # Config file
    DeclareLaunchArgument(name="config_file", default_value="config_ufpr_map.yaml", description="Name of the config file"),
]

def launch_setup(context):
    config_file_path = PathJoinSubstitution([CONFIGS_FOLDER, LaunchConfiguration("config_file")])
    bag_path = PathJoinSubstitution([BAG_FOLDER, LaunchConfiguration("bagfile")])

    return [
        # Launch drivers
        Node(
            package='drivers',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("camera_enable"), NotSubstitution(LaunchConfiguration("bag")))),
            executable='camera_publisher',
            parameters=[{'config_file' : config_file_path}],
        ),
        Node(
            package='drivers',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("imu_enable"), NotSubstitution(LaunchConfiguration("bag")))),
            executable='bno_publisher.py',
            parameters=[{'config_file' : config_file_path}],
        ),

        # Launch VO related nodes
        Node(   
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("vo_enable")),
            executable='stereo_vo_node',
            parameters=[{'config_file' : config_file_path}],
        ),
        Node(
            package='stereo_vo',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("vo_enable"), LaunchConfiguration("estimate_depth"))),
            executable='depth_estimator_server',
            parameters=[{'config_file' : config_file_path}],
        ),
        Node(
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("vo_enable")),
            executable='feature_extractor_server',
            parameters=[{'config_file' : config_file_path}],
        ),

        # Launch bag file
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("bag")),
            cmd=["ros2", "bag", "play", bag_path],
            output="screen"
        )
    ]

def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld