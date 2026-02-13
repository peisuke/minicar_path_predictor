#!/usr/bin/env python3
"""Launch ML-based navigation node.

Compatible with minicar_navigation launch parameters.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Default config: ~/ml_controllers.yaml (tunable), fallback to package config
    default_config = os.path.expanduser('~/ml_controllers.yaml')
    if not os.path.exists(default_config):
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config = os.path.join(pkg_path, 'config', 'ml_nav.yaml')

    # Control parameters (from config file)
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_config,
        description='Path to params YAML for ml_nav_node'
    )

    # Model path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/ubuntu/ros2_ws/src/minicar_path_predictor/data/models/angle_predictor.pt',
        description='Path to trained model'
    )

    # Namespace configuration (matching minicar_navigation)
    sim_ns = DeclareLaunchArgument('sim_ns', default_value='sim_robot')
    real_ns = DeclareLaunchArgument('real_ns', default_value='real_robot')
    input_sim = DeclareLaunchArgument('input_sim', default_value='true')
    input_real = DeclareLaunchArgument('input_real', default_value='false')
    output_sim = DeclareLaunchArgument('output_sim', default_value='true')
    output_real = DeclareLaunchArgument('output_real', default_value='false')
    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')
    robot_type = DeclareLaunchArgument(
        'robot_type',
        default_value='diff',
        description="Robot type: 'diff' for diff_drive or 'ackermann' for ackermann steering"
    )

    # ML Navigation Node
    ml_nav_node = Node(
        package='minicar_path_predictor',
        executable='ml_nav_node',
        name='ml_nav_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                # Override with launch arguments
                'model_path': LaunchConfiguration('model_path'),
                'sim_ns': LaunchConfiguration('sim_ns'),
                'real_ns': LaunchConfiguration('real_ns'),
                'input_sim': LaunchConfiguration('input_sim'),
                'input_real': LaunchConfiguration('input_real'),
                'output_sim': LaunchConfiguration('output_sim'),
                'output_real': LaunchConfiguration('output_real'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_type': LaunchConfiguration('robot_type'),
            },
        ],
    )

    # Parameter server node for dynamic parameter management
    # Reuses minicar_navigation's param_server_node
    param_server_node = Node(
        package='minicar_navigation',
        executable='param_server_node.py',
        name='param_server_node',
        output='screen',
        parameters=[{
            'config_path': LaunchConfiguration('params_file'),
            'target_node': '/ml_nav_node',
        }],
    )

    return LaunchDescription([
        params_file,
        model_path_arg,
        sim_ns, real_ns,
        input_sim, input_real,
        output_sim, output_real,
        use_sim_time,
        robot_type,
        ml_nav_node,
        param_server_node,
    ])
