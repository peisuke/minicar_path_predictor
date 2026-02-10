#!/bin/bash
# Run ML navigation node with system Python (required for ROS2 + torch compatibility)

export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash

MODEL_PATH="${1:-/home/ubuntu/ros2_ws/src/minicar_path_predictor/data/models/angle_predictor.pt}"

ros2 run minicar_path_predictor ml_nav_node --ros-args \
    -p model_path:="$MODEL_PATH" \
    -p lookahead_distance:=0.5 \
    -p target_velocity:=0.3 \
    -p max_angular_velocity:=1.0 \
    -p kp_angular:=2.0 \
    -p kd_angular:=0.5 \
    -p control_rate:=10.0
