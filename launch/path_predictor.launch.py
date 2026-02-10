"""Launch file for ML-based path predictor node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='weights/best_model.pth',
        description='Path to trained model checkpoint'
    )

    model_info_path_arg = DeclareLaunchArgument(
        'model_info_path',
        default_value='',
        description='Path to model_info.json (auto-detected if empty)'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='auto',
        description='Device: cuda, cpu, or auto'
    )

    sim_ns_arg = DeclareLaunchArgument(
        'sim_ns',
        default_value='sim_robot',
        description='Simulation robot namespace'
    )

    real_ns_arg = DeclareLaunchArgument(
        'real_ns',
        default_value='robot',
        description='Real robot namespace'
    )

    is_simulation_arg = DeclareLaunchArgument(
        'is_simulation',
        default_value='true',
        description='Whether running in simulation'
    )

    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='ackermann',
        description='Robot type: ackermann or diff'
    )

    max_linear_vel_arg = DeclareLaunchArgument(
        'max_linear_vel',
        default_value='1.0',
        description='Maximum linear velocity (m/s)'
    )

    max_angular_vel_arg = DeclareLaunchArgument(
        'max_angular_vel',
        default_value='1.5',
        description='Maximum angular velocity (rad/s)'
    )

    # Create node
    path_predictor_node = Node(
        package='minicar_path_predictor',
        executable='inference_node',
        name='path_predictor_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'model_info_path': LaunchConfiguration('model_info_path'),
            'device': LaunchConfiguration('device'),
            'sim_ns': LaunchConfiguration('sim_ns'),
            'real_ns': LaunchConfiguration('real_ns'),
            'is_simulation': LaunchConfiguration('is_simulation'),
            'robot_type': LaunchConfiguration('robot_type'),
            'max_linear_vel': LaunchConfiguration('max_linear_vel'),
            'max_angular_vel': LaunchConfiguration('max_angular_vel'),
        }]
    )

    return LaunchDescription([
        model_path_arg,
        model_info_path_arg,
        device_arg,
        sim_ns_arg,
        real_ns_arg,
        is_simulation_arg,
        robot_type_arg,
        max_linear_vel_arg,
        max_angular_vel_arg,
        path_predictor_node,
    ])
