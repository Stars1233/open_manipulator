# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Daeyeol Kang

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():  
    # Parameters
    drawing_height_arg = DeclareLaunchArgument('drawing_height', default_value='0.025')
    smoothing_sigma_arg = DeclareLaunchArgument('smoothing_sigma', default_value='1.0')
    resample_num_pts_arg = DeclareLaunchArgument('resample_num_pts', default_value='100')
    home_x_arg = DeclareLaunchArgument('home_x', default_value='0.124')
    home_y_arg = DeclareLaunchArgument('home_y', default_value='0.0')
    home_z_arg = DeclareLaunchArgument('home_z', default_value='0.081')
    approach_duration_arg = DeclareLaunchArgument('approach_duration', default_value='2.0')
    home_duration_arg = DeclareLaunchArgument('home_duration', default_value='4.0')
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(get_package_share_directory('open_manipulator_playground'), 'images', 'square.png')
    )
    joint5_angle_arg = DeclareLaunchArgument('joint5_angle', default_value='90.0')
    hover_height_arg = DeclareLaunchArgument('hover_height', default_value='0.08')
    enable_debug_view_arg = DeclareLaunchArgument('enable_debug_view', default_value='false')

    # shape detector node
    shape_detector = Node(
        package='open_manipulator_playground',
        executable='shape_detector_node.py',
        name='shape_detector_node',
        output='screen',
        parameters=[
            {'image_path': LaunchConfiguration('image_path')},
            {'trajectory_topic': '/drawing_trajectory'},
            {'workspace_x_min': 0.10}, 
            {'workspace_x_max': 0.26}, 
            {'workspace_y_min': -0.13}, 
            {'workspace_y_max': 0.13},  
            {'workspace_z': LaunchConfiguration('drawing_height')},
            {'smoothing_sigma': LaunchConfiguration('smoothing_sigma')},
            {'resample_num_pts': LaunchConfiguration('resample_num_pts')},
            {'enable_debug_view': LaunchConfiguration('enable_debug_view')}
        ]
    )
    
    # trajectory controller node
    omx_controller = Node(
        package='open_manipulator_playground',
        executable='omx_trajectory_controller_node.py',
        name='omx_trajectory_controller_node',
        output='screen',
        parameters=[
            {'trajectory_topic': '/drawing_trajectory'},
            {'drawing_z': LaunchConfiguration('drawing_height')},
            {'hover_height': LaunchConfiguration('hover_height')},
            {'home_x': LaunchConfiguration('home_x')},
            {'home_y': LaunchConfiguration('home_y')},
            {'home_z': LaunchConfiguration('home_z')},
            {'approach_duration': LaunchConfiguration('approach_duration')},
            {'home_duration': LaunchConfiguration('home_duration')},
            {'joint5_angle': LaunchConfiguration('joint5_angle')}
        ]
    )
    
    return LaunchDescription([
        drawing_height_arg,
        smoothing_sigma_arg,
        resample_num_pts_arg,
        home_x_arg,
        home_y_arg,
        home_z_arg,
        approach_duration_arg,
        home_duration_arg,
        image_path_arg,
        joint5_angle_arg,
        hover_height_arg,
        enable_debug_view_arg,
        shape_detector,
        omx_controller
    ])
