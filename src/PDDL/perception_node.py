#!/usr/bin/env python3
"""
ROS Node: Perception Pipeline with Real Odometry Integration

Subscribes to:
- /drone/gt_pose: Ground truth pose (Odometry)
- /drone/front/image_raw: Camera images

Processes flight_plan.json and updates it with 3D coordinates
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
import sys

# Import the perception pipeline
from perception_pipeline import (
    PerceptionPipeline,
    DronePose,
    CameraIntrinsics
)


class PerceptionNode(Node):
    """
    ROS 2 Node for perception pipeline with real odometry data
    """
    
    def __init__(self):
        super().__init__('perception_node')
        
        # Initialize perception pipeline
        self.camera = CameraIntrinsics.default_drone_camera()
        self.pipeline = PerceptionPipeline(camera_intrinsics=self.camera)
        
        # Current drone pose (updated from /gt_pose)
        self.current_pose = DronePose(
            x=0.0, y=0.0, z=1.5,
            roll=0.0, pitch=0.0, yaw=0.0
        )
        
        # Latest camera image
        self.latest_image = None
        self.bridge = CvBridge()
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            Pose,
            '/drone/gt_pose',
            self.pose_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/drone/front/image_raw',
            self.image_callback,
            10
        )
        
        # Flight plan path
        self.flight_plan_path = "/home/psp/ros_ws/src/vpdrones/src/sjtu_drone/vpdrones/flight_plan.json"
        
        self.get_logger().info("Perception Node initialized")
        self.get_logger().info(f"Subscribing to /drone/gt_pose for pose")
        self.get_logger().info(f"Flight plan: {self.flight_plan_path}")
        
        # One-shot timer to process after callbacks have had time to fire
        self.startup_timer = self.create_timer(3.0, self.startup_callback)
    
    def startup_callback(self):
        """Called once after 3s to allow subscriptions to receive data."""
        self.startup_timer.cancel()
        self.process_flight_plan()
    
    def pose_callback(self, msg: Pose):
        """
        Callback for /drone/gt_pose topic
        Updates current drone pose
        """
        # Extract position
        pos = msg.position
        
        # Extract orientation (quaternion to euler)
        q = msg.orientation
        
        # Convert quaternion to euler angles (ZYX convention)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        self.current_pose = DronePose(
            x=pos.x,
            y=pos.y,
            z=pos.z,
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )
        
        self.get_logger().debug(
            f"Pose updated: pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}), "
            f"yaw={yaw:.2f}"
        )
    
    def image_callback(self, msg: Image):
        """
        Callback for camera image topic
        Stores latest image for processing
        """
        try:
            # Convert ROS Image to OpenCV format
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def process_flight_plan(self):
        """
        Process the flight plan JSON file with current pose and image
        """
        if self.latest_image is None:
            self.get_logger().warn("No camera image available yet")
            return
        
        try:
            self.get_logger().info("Processing flight plan...")
            
            # Process flight plan with current data
            updated_plan = self.pipeline.process_flight_plan(
                flight_plan_path=self.flight_plan_path,
                rgb_image=self.latest_image,
                drone_pose=self.current_pose
            )
            
            self.get_logger().info(
                f"✓ Flight plan updated with {len(updated_plan.get('objects', []))} objects"
            )
            
            # Log object coordinates
            for obj in updated_plan.get("objects", []):
                if 'estimated_coords' in obj:
                    coords = obj['estimated_coords']
                    self.get_logger().info(
                        f"  {obj['id']}: ({coords[0]}, {coords[1]}, {coords[2]}) m"
                    )
        
        except Exception as e:
            self.get_logger().error(f"Failed to process flight plan: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    perception_node = PerceptionNode()
    
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
