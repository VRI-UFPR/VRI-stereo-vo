#!/bin/bash
#set -e

function ros_source_env() 
{
	if [ -f "$1" ]; then
		echo "sourcing   $1"
		source "$1"
    fi
}

if [[ "$ROS_DISTRO" == "melodic" || "$ROS_DISTRO" == "noetic" ]]; then
	ros_source_env "/opt/ros/$ROS_DISTRO/setup.bash"
else
	ros_source_env "$ROS_ROOT/install/setup.bash"
fi

echo "ROS_DISTRO $ROS_DISTRO"
echo "ROS_ROOT   $ROS_ROOT"

export AMENT_PREFIX_PATH=${AMENT_PREFIX_PATH}:${CMAKE_PREFIX_PATH}
exec "$@"