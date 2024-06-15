from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
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
]

def launch_setup(context):
    return [
        # Launch drivers
        Node(
            package='drivers',
            condition=IfCondition(LaunchConfiguration("camera_enable")),
            executable='camera_publisher',
        ),
        Node(
            package='drivers',
            condition=IfCondition(LaunchConfiguration("imu_enable")),
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
    ]

def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld