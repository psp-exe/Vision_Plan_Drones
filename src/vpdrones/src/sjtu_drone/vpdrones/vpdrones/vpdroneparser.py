import rclpy
from rclpy.node import Node
import json
import time
import threading
import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

# --- CONFIGURATION ---
JSON_FILE_PATH = "/home/psp/ros_ws/src/vpdrones/src/sjtu_drone/vpdrones/flight_plan.json"
TARGET_IMG_PATH = "/home/psp/ros_ws/src/vpdrones/src/sjtu_drone/imgs/target.png"
CAMERA_TOPIC = "/drone/front/image_raw" # Check 'ros2 topic list' if this fails
VEL_TOPIC = "/drone/cmd_vel"

class VPDroneParser(Node):
    def __init__(self):
        super().__init__('vpdroneparser')

        # -- PUBLISHERS --
        self.velocity_publisher = self.create_publisher(Twist, VEL_TOPIC, 10)
        self.takeoff_publisher = self.create_publisher(Empty, '/drone/takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, '/drone/land', 10)
        
        # -- SUBSCRIBERS --
        # We subscribe to the camera but only process when needed
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10)

        # -- COMPUTER VISION SETUP --
        self.following_active = False # Flag to turn on/off vision control
        self.target_reached = False
        self.load_target_image()

        # -- INPUT THREAD --
        self.get_logger().info("Vision Drone Ready. Press 'Enter' to start mission.")
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

    def load_target_image(self):
        """Loads the reference image for feature matching."""
        self.ref_img = cv2.imread(TARGET_IMG_PATH, 0) # Load as grayscale
        if self.ref_img is None:
            self.get_logger().error(f"Could not load {TARGET_IMG_PATH}. Check path!")
            return
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_img, None)
        self.get_logger().info(f"Target loaded. Keypoints: {len(self.kp_ref)}")

    def wait_for_input(self):
        while rclpy.ok():
            try:
                input()
                self.get_logger().info("Starting Mission...")
                self.execute_mission()
            except EOFError:
                break

    def execute_mission(self):
        try:
            with open("/home/psp/ros_ws/src/vpdrones/src/sjtu_drone/vpdrones/flight_plan.json", 'r') as f:
                commands = json.load(f)
            if isinstance(commands, dict): commands = [commands]

            for command in commands:
                self.process_command(command)
                time.sleep(1.0) # Stability pause
            
            self.get_logger().info("Mission Complete.") 

        except Exception as e:
            self.get_logger().error(f"Error reading JSON: {e}")

    def process_command(self, command):
        action = command.get('action', '').lower()
        self.get_logger().info(f">>> Executing: {action}")

        if action == 'takeoff':
            self.takeoff_publisher.publish(Empty())
            time.sleep(4)
            
        elif action == 'land':
            self.land_publisher.publish(Empty())
            time.sleep(4)

        elif action == 'fly':
            self.move_drone(command.get('direction', 'forward'), 
                            float(command.get('distance', 1.0)), 
                            float(command.get('speed', 0.5)))

        elif action == 'rotate':
            self.rotate_drone(float(command.get('angle', 90.0)))

        elif action == 'follow_target':
            # Enable Vision Loop and BLOCK here until it finishes
            if self.ref_img is None:
                self.get_logger().warn("Skipping follow_target: No image loaded.")
                return

            self.target_reached = False
            self.following_active = True
            
            # Wait loop: The image_callback is now driving the drone.
            # We wait here until the callback sets target_reached to True.
            start_wait = time.time()
            while not self.target_reached and rclpy.ok():
                time.sleep(0.1)
                # Timeout safety (e.g. 30 seconds max search)
                if time.time() - start_wait > 30.0:
                    self.get_logger().warn("Target follow timed out.")
                    break
            
            self.following_active = False
            self.stop_drone()

    def image_callback(self, msg):
        """
        Main Vision Control Loop with RANSAC Outlier Rejection
        """
        if not self.following_active:
            return

        try:
            # 1. Image Conversion
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w, _ = frame.shape

            # 2. Detect Features
            kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

            if des_frame is None:
                return

            # 3. Match Features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.des_ref, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Keep top 30 matches to have enough data for RANSAC
            good_matches = matches[:30]

            if len(good_matches) > 10:
                # Prepare points for RANSAC
                src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # 4. RANSAC (The Fix!) - Find the Perspective Transform
                # This identifies which points actually form a valid object shape
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    matchesMask = mask.ravel().tolist()
                    
                    # Filter only the valid points (inliers)
                    valid_dst_pts = dst_pts[mask.ravel() == 1]
                    
                    # If we have enough valid points after filtering
                    if len(valid_dst_pts) > 5:
                        # Calculate Bounding Box of ONLY valid points
                        x_min = np.min(valid_dst_pts[:,0,0])
                        x_max = np.max(valid_dst_pts[:,0,0])
                        y_min = np.min(valid_dst_pts[:,0,1])
                        y_max = np.max(valid_dst_pts[:,0,1])

                        # Calculate Center and Area
                        avg_x = (x_min + x_max) / 2
                        obj_width = x_max - x_min
                        obj_height = y_max - y_min
                        obj_area = obj_width * obj_height
                        
                        # Draw the bounding box for debugging (Blue Box)
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 3)

                        # Calculate Error and Ratio
                        err_x = (w / 2) - avg_x
                        screen_area = w * h
                        
                        # Call the controller
                        self.visual_servo_control(err_x, obj_area, screen_area)
                    else:
                        self.get_logger().info("RANSAC rejected matches (bad geometry).")
                        self.stop_drone()
                else:
                    self.get_logger().info("Could not find Homography.")
                    self.stop_drone()
            else:
                self.stop_drone()

            # Show Debug Window
            cv2.imshow("Drone Vision", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"CV Error: {e}")

    def visual_servo_control(self, err_x, obj_area, screen_area):
        msg = Twist()
        
        # Calculate current ratio
        current_ratio = obj_area / screen_area
        target_ratio = 0.15  # Stop when object is 15% of screen
        
        # LOGGING (Crucial for debugging)
        # Check your terminal for these numbers!
        self.get_logger().info(f"Ratio: {current_ratio:.4f} / {target_ratio} | Error X: {err_x:.1f}")

        if current_ratio < target_ratio:
            # P-Controller for Rotation
            Kp_rot = 0.002
            msg.angular.z = Kp_rot * err_x
            
            # Move Forward
            msg.linear.x = 0.2 
            self.velocity_publisher.publish(msg)
            self.target_reached = False
        else:
            self.get_logger().info("Target Reached (Size Condition Met)!")
            self.target_reached = True
            self.stop_drone()

    def move_drone(self, direction, distance, speed):
        # ... (Same as your previous successful code) ...
        msg = Twist()
        if direction == 'forward': msg.linear.x = speed
        elif direction == 'backward': msg.linear.x = -speed
        elif direction == 'left': msg.linear.y = speed
        elif direction == 'right': msg.linear.y = -speed
        elif direction == 'up': msg.linear.z = speed
        elif direction == 'down': msg.linear.z = -speed
        
        duration = distance / speed
        start = time.time()
        while time.time() - start < duration:
            self.velocity_publisher.publish(msg)
            time.sleep(0.1)
        self.stop_drone()

    def rotate_drone(self, angle):
        # ... (Same as your previous successful code) ...
        msg = Twist()
        ang_speed = 0.5
        target_rad = (abs(angle) * 3.14159) / 180.0
        msg.angular.z = ang_speed if angle > 0 else -ang_speed
        
        duration = target_rad / ang_speed
        start = time.time()
        while time.time() - start < duration:
            self.velocity_publisher.publish(msg)
            time.sleep(0.1)
        self.stop_drone()

    def stop_drone(self):
        self.velocity_publisher.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = VPDroneParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()