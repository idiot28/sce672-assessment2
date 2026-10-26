#!/usr/bin/env python3
"""
flatness_pursuit.launch.py - ROS 2 Launch File for rm_flatness_nav (v4)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("conn_type", default_value="ap"),
        DeclareLaunchArgument("target_tag_id", default_value="3"),
        DeclareLaunchArgument("use_sim", default_value="false"),
        DeclareLaunchArgument("v_max_mps", default_value="0.38"),
        DeclareLaunchArgument("standoff_m", default_value="0.20"),
        DeclareLaunchArgument("arm_x_mm", default_value="140"),
        DeclareLaunchArgument("arm_y_mm", default_value="35"),
        DeclareLaunchArgument("auto_exit", default_value="false"),
        DeclareLaunchArgument("output_dir", default_value="/mnt/d/phd/sce672 - advanced robotics/project apriltag v4"),
        Node(
            package="rm_flatness_nav",
            executable="mission_node",
            name="rm_flatness_nav_v4",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "conn_type": LaunchConfiguration("conn_type"),
                "target_tag_id": LaunchConfiguration("target_tag_id"),
                "use_sim": LaunchConfiguration("use_sim"),
                "v_max_mps": LaunchConfiguration("v_max_mps"),
                "standoff_m": LaunchConfiguration("standoff_m"),
                "arm_x_mm": LaunchConfiguration("arm_x_mm"),
                "arm_y_mm": LaunchConfiguration("arm_y_mm"),
                "auto_exit": LaunchConfiguration("auto_exit"),
                "output_dir": LaunchConfiguration("output_dir"),
            }],
        ),
    ])
