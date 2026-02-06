import rclpy
from rclpy.node import Node
import threading
import cv2
import paramiko
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scp import SCPClient
import os

# --- SSH CONFIGURATION (EDIT THIS) ---
REMOTE_HOST = '100.88.111.51'     # IP of your powerful server
REMOTE_USER = 'dartagnan-dev'             # Username on server
REMOTE_PASS = 'dartagnan-dev'         # Password
REMOTE_DIR  = '/home/dartagnan-dev/pranav_ws/chat_response/' # Folder on server

# --- LOCAL FILE PATHS ---
LOCAL_IMG_PATH = "/tmp/drone_view.jpg"
LOCAL_TXT_PATH = "/tmp/query.txt"
LOCAL_JSON_PATH = "/home/psp/ros_ws/src/vpdrones/src/sjtu_drone/vpdrones/flight_plan.json"
REMOTE_JSON_NAME = "flight_plan.json"

class RemoteBrainNode(Node):
    def __init__(self):
        super().__init__('remote_brain_node')
        
        # ROS Setup
        self.bridge = CvBridge()
        self.latest_image = None
        
        # Only Subscribe to Camera
        self.image_sub = self.create_subscription(
            Image, '/drone/front/image_raw', self.image_callback, 10)
        
        # SSH Client Setup
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.get_logger().info("Remote Brain (Data Only) Initialized.")
        self.get_logger().info(f"Target Server: {REMOTE_USER}@{REMOTE_HOST}")
        self.get_logger().info("Type your query and press Enter to Sync.")

        # Start Input Thread
        self.input_thread = threading.Thread(target=self.chatbot_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

    def image_callback(self, msg):
        """Continuously updates the latest image from drone camera"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def chatbot_loop(self):
        """Main interaction loop"""
        while rclpy.ok():
            # 1. Get User Input
            try:
                prompt = input("\nUSER: ")
            except EOFError:
                break

            if not prompt: continue

            # 2. Capture and Save Data locally
            if self.latest_image is not None:
                cv2.imwrite(LOCAL_IMG_PATH, self.latest_image)
                with open(LOCAL_TXT_PATH, 'w') as f:
                    f.write(prompt)
                
                self.get_logger().info(f"Captured image and text. Connecting to {REMOTE_HOST}...")
                
                # 3. Perform SSH Operations
                success = self.communicate_with_server()
                
                if success:
                    self.get_logger().info("SUCCESS: JSON file copied back to " + LOCAL_JSON_PATH)
                    # Verify content (Optional)
                    with open(LOCAL_JSON_PATH, 'r') as f:
                        print(f"Content received: {f.read()}")
                else:
                    self.get_logger().error("FAILED: Could not complete transfer.")
            else:
                self.get_logger().warn("No image received yet. Is the simulation running?")

    def communicate_with_server(self):
        """Uploads Image/Text and Downloads JSON"""
        try:
            self.ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASS)
            
            with SCPClient(self.ssh.get_transport()) as scp:
                # A. Upload Query and Image
                self.get_logger().info("Uploading query.txt and input.jpg...")
                scp.put(LOCAL_TXT_PATH, remote_path=REMOTE_DIR + "query.txt")
                scp.put(LOCAL_IMG_PATH, remote_path=REMOTE_DIR + "input.jpg")
                
                # B. Download the Resulting JSON
                self.get_logger().info(f"Downloading {REMOTE_JSON_NAME}...")
                remote_json_full_path = os.path.join(REMOTE_DIR, REMOTE_JSON_NAME)
                scp.get(remote_json_full_path, local_path=LOCAL_JSON_PATH)
            
            self.ssh.close()
            return True
            
        except Exception as e:
            self.get_logger().error(f"SSH/SCP Error: {e}")
            return False

def main(args=None): 
    rclpy.init(args=args)
    node = RemoteBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()