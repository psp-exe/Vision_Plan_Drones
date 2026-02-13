import rclpy
from rclpy.node import Node
import json
import time
import math
import threading
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from geometry_msgs.msg import Pose

# --- CONFIGURATION ---
PLAN_JSON_PATH = "/home/psp/ros_ws/src/PDDL/generated_plans/plan_latest.json"
VEL_TOPIC = "/drone/cmd_vel"
POSE_TOPIC = "/drone/gt_pose"


class VPDroneParser(Node):
    def __init__(self):
        super().__init__('vpdroneparser')

        # -- PUBLISHERS --
        self.velocity_publisher = self.create_publisher(Twist, VEL_TOPIC, 10)
        self.takeoff_publisher = self.create_publisher(Empty, '/drone/takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, '/drone/land', 10)

        # -- SUBSCRIBERS --
        self.pose_subscriber = self.create_subscription(
            Pose, POSE_TOPIC, self.pose_callback, 10)

        # -- DRONE STATE --
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
        self.pose_received = False

        # -- FLIGHT PARAMETERS --
        self.fly_speed = 0.75          # m/s cruise speed
        self.position_tolerance = 0.3 # meters — close enough to target
        self.hover_duration = 5.0     # seconds to hover during scan
        self.takeoff_wait = 5.0       # seconds to wait after takeoff
        self.land_wait = 5.0          # seconds to wait after land

        # -- INPUT THREAD --
        self.get_logger().info("PDDL Plan Executor Ready.")
        self.get_logger().info(f"Plan file: {PLAN_JSON_PATH}")
        self.get_logger().info("Press 'Enter' to execute the plan.")
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

    # =====================================================================
    # POSE CALLBACK
    # =====================================================================
    def pose_callback(self, msg):
        """Update current drone position and yaw from ground truth pose."""
        self.current_x = msg.position.x
        self.current_y = msg.position.y
        self.current_z = msg.position.z

        # Extract yaw from quaternion
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.pose_received = True

    # =====================================================================
    # PLAN LOADING
    # =====================================================================
    def load_plan(self):
        """Load the PDDL plan from JSON file."""
        try:
            with open(PLAN_JSON_PATH, 'r') as f:
                plan = json.load(f)

            if not plan.get('plan_found', False):
                self.get_logger().error("No valid plan found in JSON.")
                return None

            self.get_logger().info(f"Plan loaded: {plan['num_actions']} actions, "
                                  f"makespan {plan['makespan']}s")
            return plan

        except Exception as e:
            self.get_logger().error(f"Error loading plan: {e}")
            return None

    # =====================================================================
    # INPUT / MISSION START
    # =====================================================================
    def wait_for_input(self):
        while rclpy.ok():
            try:
                input()
                self.execute_plan()
            except EOFError:
                break

    def execute_plan(self):
        """Load and execute the PDDL plan sequentially."""
        plan = self.load_plan()
        if plan is None:
            return

        actions = plan.get('actions', [])

        # Wait for initial pose before executing any actions
        if not self.pose_received:
            self.get_logger().info("Waiting for initial pose from /drone/gt_pose...")
            timeout = time.time() + 10.0
            while not self.pose_received and time.time() < timeout:
                time.sleep(0.1)
            if not self.pose_received:
                self.get_logger().error("No pose data received, aborting plan.")
                return

        self.get_logger().info(
            f"Initial pose: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})")

        self.get_logger().info("=" * 50)
        self.get_logger().info("EXECUTING PDDL PLAN")
        self.get_logger().info("=" * 50)

        for i, action in enumerate(actions, 1):
            name = action['name']
            params = action.get('parameters', [])
            coords = action.get('coordinates', None)

            self.get_logger().info(f"\n[{i}/{len(actions)}] Action: {name}")
            self.get_logger().info(f"  Parameters: {params}")
            if coords:
                self.get_logger().info(
                    f"  Target: ({coords['x']}, {coords['y']}, {coords['z']})")

            # Dispatch to the right handler
            if name == 'take_off':
                self.action_takeoff()
            elif name == 'fly_to_target':
                if coords:
                    self.action_fly_to_target(
                        coords['x'], coords['y'], coords['z'],
                        target_name=params[1] if len(params) > 1 else "unknown")
                else:
                    self.get_logger().warn("fly_to_target missing coordinates, skipping.")
            elif name == 'scan_target':
                target_name = params[1] if len(params) > 1 else "unknown"
                self.action_scan_target(target_name)
            elif name == 'land':
                self.action_land()
            else:
                self.get_logger().warn(f"Unknown action: {name}, skipping.")

            time.sleep(1.0)  # brief pause between actions

        self.get_logger().info("=" * 50)
        self.get_logger().info("PLAN EXECUTION COMPLETE")
        self.get_logger().info("=" * 50)

    # =====================================================================
    # ACTION: take_off
    # =====================================================================
    def action_takeoff(self):
        """Publish takeoff command and wait for the drone to stabilize."""
        self.get_logger().info("  Taking off...")
        self.takeoff_publisher.publish(Empty())
        time.sleep(self.takeoff_wait)
        self.get_logger().info(f"  Airborne at z={self.current_z:.2f}m")

    # =====================================================================
    # ACTION: fly_to_target (closed-loop position control)
    # =====================================================================
    def action_fly_to_target(self, target_x, target_y, target_z,
                             target_name="target"):
        """
        Fly to a 3D coordinate using proportional velocity control.

        Continuously publishes velocity commands toward the target until
        the drone is within position_tolerance of the target coordinates.
        """
        target_x = target_x*10
        target_y = target_y*10
        target_z = target_z*10
        # Clamp target Z to minimum safe altitude
        target_z = max(target_z, 0.75)

        self.get_logger().info(
            f"  Flying to {target_name} at ({target_x}, {target_y}, {target_z})")

        # Proportional gain
        Kp = 0.5
        max_speed = self.fly_speed
        rate_hz = 10  # control loop frequency
        timeout = 60.0  # safety timeout

        start_time = time.time()

        while rclpy.ok():
            # Compute errors
            ex = target_x - self.current_x
            ey = target_y - self.current_y
            ez = target_z - self.current_z
            distance = math.sqrt(ex**2 + ey**2 + ez**2)

            # Check if arrived
            if distance < self.position_tolerance:
                self.stop_drone()
                self.get_logger().info(
                    f"  Arrived at {target_name} "
                    f"(error: {distance:.2f}m)")
                return

            # Check timeout
            if time.time() - start_time > timeout:
                self.stop_drone()
                self.get_logger().warn(
                    f"  Fly timeout after {timeout}s "
                    f"(distance remaining: {distance:.2f}m)")
                return

            # Proportional control with speed clamping
            vx = max(-max_speed, min(max_speed, Kp * ex))
            vy = max(-max_speed, min(max_speed, Kp * ey))
            vz = max(-max_speed, min(max_speed, Kp * ez))

            # Yaw alignment: face the target (only when far enough)
            xy_dist = math.sqrt(ex**2 + ey**2)
            yaw_rate = 0.0
            if xy_dist > 1.0:
                desired_yaw = math.atan2(ey, ex)
                yaw_error = desired_yaw - self.current_yaw
                # Normalize to [-pi, pi]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                yaw_rate = max(-0.3, min(0.3, 0.3 * yaw_error))

            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.linear.z = vz
            msg.angular.z = yaw_rate
            self.velocity_publisher.publish(msg)

            time.sleep(1.0 / rate_hz)

    # =====================================================================
    # ACTION: scan_target (hover for 5 seconds)
    # =====================================================================
    def action_scan_target(self, target_name="target"):
        """Hover in place for hover_duration seconds to scan the target."""
        self.get_logger().info(
            f"  Scanning {target_name} — hovering for {self.hover_duration}s...")
        self.stop_drone()
        time.sleep(self.hover_duration)
        self.get_logger().info(f"  Scan of {target_name} complete.")

    # =====================================================================
    # ACTION: land
    # =====================================================================
    def action_land(self):
        """Publish land command and wait for the drone to touch down."""
        self.get_logger().info("  Landing...")
        self.land_publisher.publish(Empty())
        time.sleep(self.land_wait)
        self.get_logger().info(f"  Landed at z={self.current_z:.2f}m")

    # =====================================================================
    # UTILITY
    # =====================================================================
    def stop_drone(self):
        """Stop all drone movement."""
        self.velocity_publisher.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = VPDroneParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()