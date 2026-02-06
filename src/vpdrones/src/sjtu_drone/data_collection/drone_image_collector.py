import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import os
from datetime import datetime
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

# --- CONFIGURATION ---
CAMERA_TOPIC = "/drone/front/image_raw"
VEL_TOPIC = "/drone/cmd_vel"
OUTPUT_DIR = "/home/psp/ros_ws/src/PDDL/vlm_training_data/images"

class DroneImageCollector(Node):
    def __init__(self):
        super().__init__('drone_image_collector')
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # -- PUBLISHERS --
        self.velocity_publisher = self.create_publisher(Twist, VEL_TOPIC, 10)
        self.takeoff_publisher = self.create_publisher(Empty, '/drone/takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, '/drone/land', 10)
        
        # -- SUBSCRIBERS --
        self.bridge = CvBridge()
        self.current_frame = None
        self.image_subscriber = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10)
        
        # Image capture counter
        self.capture_count = 0
        
        self.get_logger().info("=== Drone Image Collector Ready ===")
        self.get_logger().info(f"Output directory: {OUTPUT_DIR}")
        
        # Wait for camera to be ready
        self.wait_for_camera()
        
        self.get_logger().info("Starting autonomous data collection in 3 seconds...")
        time.sleep(3)
        
        # Start autonomous mission
        self.execute_data_collection()
    
    def image_callback(self, msg):
        """Store the latest frame"""
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
    
    def wait_for_camera(self):
        """Wait until camera starts streaming frames"""
        self.get_logger().info("Waiting for camera feed...")
        timeout = 30.0  # 30 seconds timeout
        start_time = time.time()
        
        while self.current_frame is None and rclpy.ok():
            if time.time() - start_time > timeout:
                self.get_logger().error("Camera timeout! No frames received.")
                self.get_logger().error("Check if simulation is running and camera topic is correct.")
                raise RuntimeError("Camera initialization failed")
            
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        
        self.get_logger().info("✓ Camera feed detected and ready!")
    
    def capture_and_save_image(self, position_name="unknown"):
        """
        Capture current frame and save (no grid overlay)
        """
        # Wait a bit and ensure we have a fresh frame
        for _ in range(5):  # Try up to 5 times
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_frame is not None:
                break
            time.sleep(0.1)
        
        if self.current_frame is None:
            self.get_logger().warn(f"No frame available to capture at position: {position_name}")
            return False
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drone_img_{self.capture_count:03d}_{position_name}_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save clean image (no grid overlay)
        cv2.imwrite(filepath, self.current_frame)
        self.get_logger().info(f"✓ Saved: {filename}")
        
        self.capture_count += 1
        return True
    
    def execute_data_collection(self):
        """
        Autonomous flight pattern to collect diverse images
        """
        try:
            self.get_logger().info(">>> Taking off...")
            self.takeoff_publisher.publish(Empty())
            time.sleep(5)  # Wait for stable takeoff
            
            # Define a flight pattern for diverse data collection
            # Pattern: Grid exploration with multiple altitudes
            
            flight_waypoints = [
                # Low altitude sweep
                ("center_low", "forward", 2.0, 0.3),
                ("front_low", "rotate", 90, None),
                ("front_low_rot", "right", 2.0, 0.3),
                ("right_low", "rotate", 90, None),
                ("right_low_rot", "backward", 2.0, 0.3),
                ("back_low", "rotate", 90, None),
                ("back_low_rot", "left", 2.0, 0.3),
                
                # Mid altitude
                ("center_mid", "up", 1.5, 0.3),
                ("mid_altitude", "rotate", -180, None),
                ("mid_rot", "forward", 1.5, 0.3),
                ("mid_forward", "rotate", 90, None),
                ("mid_forward_rot", "right", 1.5, 0.3),
                
                # High altitude
                ("high_pos", "up", 1.5, 0.3),
                ("high_altitude", "rotate", -90, None),
                ("high_rot", "forward", 2.0, 0.3),
                ("high_forward", "rotate", 180, None),
                
                # Return and descend
                ("return_high", "backward", 2.0, 0.3),
                ("return_mid", "down", 1.5, 0.3),
                ("return_low", "down", 1.5, 0.3),
            ]
            
            for waypoint_name, action_type, param1, param2 in flight_waypoints:
                # Capture image at this position
                self.get_logger().info(f">>> Position: {waypoint_name}")
                time.sleep(0.5)  # Stabilize
                
                # Spin once to get fresh camera frame
                rclpy.spin_once(self, timeout_sec=0.1)
                
                self.capture_and_save_image(waypoint_name)
                
                # Execute movement
                if action_type == "rotate":
                    self.rotate_drone(param1)
                else:
                    self.move_drone(action_type, param1, param2)
                
                time.sleep(0.5)  # Stabilize after movement
            
            # Final capture before landing
            self.capture_and_save_image("final_position")
            
            # Land
            self.get_logger().info(">>> Landing...")
            self.land_publisher.publish(Empty())
            time.sleep(5)
            
            self.get_logger().info(f"=== Data Collection Complete ===")
            self.get_logger().info(f"Total images captured: {self.capture_count}")
            self.get_logger().info(f"Images saved to: {OUTPUT_DIR}")
            
        except Exception as e:
            self.get_logger().error(f"Mission error: {e}")
            self.emergency_land()
    
    def move_drone(self, direction, distance, speed):
        """Move drone in specified direction"""
        msg = Twist()
        
        if direction == 'forward':
            msg.linear.x = speed
        elif direction == 'backward':
            msg.linear.x = -speed
        elif direction == 'left':
            msg.linear.y = speed
        elif direction == 'right':
            msg.linear.y = -speed
        elif direction == 'up':
            msg.linear.z = speed
        elif direction == 'down':
            msg.linear.z = -speed
        
        duration = distance / speed
        start = time.time()
        
        while time.time() - start < duration:
            self.velocity_publisher.publish(msg)
            time.sleep(0.1)
        
        self.stop_drone()
    
    def rotate_drone(self, angle):
        """Rotate drone by specified angle (degrees)"""
        msg = Twist()
        ang_speed = 0.5  # rad/s
        target_rad = (abs(angle) * 3.14159) / 180.0
        msg.angular.z = ang_speed if angle > 0 else -ang_speed
        
        duration = target_rad / ang_speed
        start = time.time()
        
        while time.time() - start < duration:
            self.velocity_publisher.publish(msg)
            time.sleep(0.1)
        
        self.stop_drone()
    
    def stop_drone(self):
        """Stop all drone movement"""
        self.velocity_publisher.publish(Twist())
        time.sleep(0.2)
    
    def emergency_land(self):
        """Emergency landing procedure"""
        self.get_logger().warn("!!! EMERGENCY LANDING !!!")
        self.stop_drone()
        self.land_publisher.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    node = DroneImageCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()