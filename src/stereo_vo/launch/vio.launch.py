from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution, AndSubstitution, NotSubstitution
from launch_ros.actions import Node

launch_args = [
    DeclareLaunchArgument(
        name="camera_enable", default_value="true", description="enable camera node"
    ),
    DeclareLaunchArgument(
        name="imu_enable", default_value="true", description="enable IMU node"
    ),
    DeclareLaunchArgument(
        name="vo_enable", default_value="true", description="enable VO pose estimation"
    ),

    DeclareLaunchArgument(
        name="bag", default_value="true", description="enable bag file"
    ),
    DeclareLaunchArgument(
        name="bagfile", default_value="Data/bags/MH_01_easy_ros2", description="Path to the bag folder to play"
    ),
]

def launch_setup(context):
    return [
        # Launch drivers
        Node(
            package='drivers',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("camera_enable"), NotSubstitution(LaunchConfiguration("bag")))),
            executable='camera_publisher',
        ),
        Node(
            package='drivers',
            condition=IfCondition(AndSubstitution(LaunchConfiguration("imu_enable"), NotSubstitution(LaunchConfiguration("bag")))),
            executable='bno_publisher.py',
        ),

        # Launch VO related nodes
        Node(   
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("vo_enable")),
            executable='stereo_vo_node',
        ),
        Node(
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("vo_enable")),
            executable='depth_estimator_server',
        ),
        Node(
            package='stereo_vo',
            condition=IfCondition(LaunchConfiguration("vo_enable")),
            executable='feature_extractor_server',
        ),

        # Launch bag file
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("bag")),
            cmd=["ros2", "bag", "play", LaunchConfiguration("bagfile")],
            output="screen"
        )
    ]

def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld