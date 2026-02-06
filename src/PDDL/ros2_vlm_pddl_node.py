#!/usr/bin/env python3
"""
=============================================================================
ROS 2 VLM PDDL Node
=============================================================================
Integrates the VLM perception pipeline with ROS 2 and PlanSys2.

Implements Section 6 of the design document:
- Subscribes to drone camera images
- Runs VLM inference to detect objects
- Computes 3D coordinates via depth estimation
- Updates PlanSys2 Knowledge Base
- Triggers planning and execution
=============================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DetectedTarget:
    """Target detected by VLM"""
    id: str
    bbox: List[int]
    world_coords: List[float]


class VLMPDDLNode(Node):
    """
    ROS 2 Node that bridges VLM perception with PlanSys2 planning.
    
    Workflow (Section 6.2):
    1. User sends command via /vlm_pddl/instruction topic
    2. Node captures image from drone camera
    3. Runs VLM inference to detect targets
    4. Computes 3D coordinates via depth estimation
    5. Updates PlanSys2 Knowledge Base via services
    6. Sets goal and triggers planning
    """
    
    def __init__(self):
        super().__init__('vlm_pddl_node')
        
        # Parameters
        self.declare_parameter('camera_topic', '/drone/front/image_raw')
        self.declare_parameter('pose_topic', '/drone/gt_pose')
        self.declare_parameter('domain_file', '/home/psp/ros_ws/src/PDDL/warehouse_drone_domain.pddl')
        self.declare_parameter('use_vlm_inference', False)  # False = simulated
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # State
        self.current_image: Optional[np.ndarray] = None
        self.current_pose: Optional[Dict] = None
        self.detected_targets: List[DetectedTarget] = []
        
        # Subscribers
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10
        )
        
        self.instruction_sub = self.create_subscription(
            String,
            '/vlm_pddl/instruction',
            self.instruction_callback,
            10
        )
        
        # Publishers
        self.pddl_problem_pub = self.create_publisher(
            String,
            '/vlm_pddl/generated_problem',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vlm_pddl/status',
            10
        )
        
        self.get_logger().info('VLM PDDL Node initialized')
        self.get_logger().info(f'  Camera topic: {camera_topic}')
        self.get_logger().info(f'  Pose topic: {pose_topic}')
        self.get_logger().info('  Waiting for instructions on /vlm_pddl/instruction')
        
    def image_callback(self, msg: Image):
        """Store latest camera image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
    
    def pose_callback(self, msg: PoseStamped):
        """Store latest drone pose"""
        self.current_pose = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'z': msg.pose.position.z,
            'qx': msg.pose.orientation.x,
            'qy': msg.pose.orientation.y,
            'qz': msg.pose.orientation.z,
            'qw': msg.pose.orientation.w
        }
    
    def instruction_callback(self, msg: String):
        """
        Process natural language instruction.
        
        This triggers the full pipeline:
        1. VLM inference
        2. Depth estimation
        3. PDDL generation
        4. PlanSys2 update
        """
        instruction = msg.data
        self.get_logger().info(f'Received instruction: "{instruction}"')
        
        self.publish_status(f'Processing instruction: {instruction}')
        
        # Check prerequisites
        if self.current_image is None:
            self.get_logger().warn('No image available - waiting for camera')
            self.publish_status('Warning: No camera image available')
            return
        
        if self.current_pose is None:
            self.get_logger().warn('No pose available - using default')
            self.current_pose = {'x': 0, 'y': 0, 'z': 1.5}
        
        # Run perception pipeline
        use_vlm = self.get_parameter('use_vlm_inference').get_parameter_value().bool_value
        
        if use_vlm:
            # Real VLM inference
            vlm_output = self.run_vlm_inference(self.current_image, instruction)
        else:
            # Simulated output for testing
            vlm_output = self.simulate_vlm_output(instruction)
        
        self.get_logger().info(f'VLM detected {len(vlm_output["objects"])} objects')
        
        # Generate PDDL problem
        pddl_problem = self.generate_pddl_problem(vlm_output)
        
        # Publish PDDL problem
        pddl_msg = String()
        pddl_msg.data = pddl_problem
        self.pddl_problem_pub.publish(pddl_msg)
        
        self.get_logger().info('PDDL problem published to /vlm_pddl/generated_problem')
        
        # Update PlanSys2 (if available)
        self.update_plansys2(vlm_output)
        
        self.publish_status('PDDL problem generated - ready for planning')
    
    def run_vlm_inference(
        self, 
        image: np.ndarray, 
        instruction: str
    ) -> Dict:
        """
        Run actual VLM inference.
        
        Replace with your fine-tuned model:
        - PaliGemma
        - LLaVA
        - Other VLM
        """
        # This would be replaced with actual model inference
        # Example using transformers:
        '''
        from transformers import AutoProcessor, AutoModel
        
        processor = AutoProcessor.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path)
        
        inputs = processor(
            images=image,
            text=f"Instruction: {instruction}\\nGenerate PDDL JSON:",
            return_tensors="pt"
        )
        
        outputs = model.generate(**inputs, max_new_tokens=512)
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        return json.loads(result)
        '''
        
        # For now, return simulated output
        return self.simulate_vlm_output(instruction)
    
    def simulate_vlm_output(self, instruction: str) -> Dict:
        """
        Simulate VLM output for testing.
        
        Parses simple instructions and generates plausible detections.
        """
        # Parse instruction for object type and color
        instruction_lower = instruction.lower()
        
        # Default object
        obj_id = "target_01"
        color = "unknown"
        category = "object"
        
        # Simple parsing
        colors = ["red", "blue", "green", "yellow", "white", "black"]
        categories = ["box", "crate", "pallet", "barrel", "container", "package"]
        
        for c in colors:
            if c in instruction_lower:
                color = c
                break
        
        for cat in categories:
            if cat in instruction_lower:
                category = cat
                break
        
        obj_id = f"{color}_{category}"
        
        # Simulated coordinates (based on current pose)
        x = self.current_pose.get('x', 0) + 5.0
        y = self.current_pose.get('y', 0) + 1.0
        z = 2.5
        
        return {
            "instruction": instruction,
            "objects": [{
                "id": obj_id,
                "type": "target",
                "bbox": [280, 200, 120, 100],
                "estimated_coords": [x, y, z]
            }],
            "goal": f"(scanned {obj_id})"
        }
    
    def generate_pddl_problem(self, vlm_output: Dict) -> str:
        """
        Generate PDDL problem file from VLM output.
        
        Uses the VLM PDDL Generator module.
        """
        from datetime import datetime
        import math
        
        drone_pos = (
            self.current_pose.get('x', 0),
            self.current_pose.get('y', 0),
            self.current_pose.get('z', 1.5)
        )
        
        lines = []
        
        # Header
        lines.append(f";; Auto-generated by VLM PDDL Node")
        lines.append(f";; Generated: {datetime.now().isoformat()}")
        lines.append("")
        
        # Problem definition
        problem_name = f"vlm_problem_{datetime.now().strftime('%H%M%S')}"
        lines.append(f"(define (problem {problem_name})")
        lines.append("  (:domain warehouse-drone)")
        lines.append("")
        
        # Objects
        lines.append("  (:objects")
        lines.append("    drone1 - drone")
        for obj in vlm_output.get("objects", []):
            lines.append(f"    {obj['id']} - target")
        lines.append("    charging_station - zone")
        lines.append("  )")
        lines.append("")
        
        # Initial state
        lines.append("  (:init")
        lines.append("    (landed drone1)")
        lines.append("    (calibrated drone1)")
        lines.append("    (docked drone1)")
        lines.append("    (at-zone drone1 charging_station)")
        lines.append("")
        lines.append(f"    (= (x drone1) {drone_pos[0]})")
        lines.append(f"    (= (y drone1) {drone_pos[1]})")
        lines.append(f"    (= (z drone1) {drone_pos[2]})")
        lines.append("")
        lines.append("    (= (battery-level drone1) 95.0)")
        lines.append("    (= (battery-capacity drone1) 100.0)")
        lines.append("    (= (discharge-rate-fly drone1) 0.5)")
        lines.append("    (= (discharge-rate-hover drone1) 0.2)")
        lines.append("    (= (recharge-rate drone1) 1.0)")
        lines.append("    (= (fly-speed drone1) 2.0)")
        lines.append("")
        lines.append("    (= (scan-range) 2.0)")
        lines.append("    (= (min-battery) 20.0)")
        lines.append("    (= (takeoff-altitude) 1.5)")
        lines.append("")
        
        # Target coordinates
        for obj in vlm_output.get("objects", []):
            coords = obj.get("estimated_coords", [0, 0, 0])
            lines.append(f"    ;; Target: {obj['id']}")
            lines.append(f"    (= (tx {obj['id']}) {coords[0]})")
            lines.append(f"    (= (ty {obj['id']}) {coords[1]})")
            lines.append(f"    (= (tz {obj['id']}) {coords[2]})")
            
            # Euclidean distance
            dist = math.sqrt(
                (coords[0] - drone_pos[0])**2 +
                (coords[1] - drone_pos[1])**2 +
                (coords[2] - drone_pos[2])**2
            )
            lines.append(f"    (= (distance-to drone1 {obj['id']}) {dist:.2f})")
            lines.append("")
        
        # Zone
        lines.append("    (= (zone-x charging_station) 0.0)")
        lines.append("    (= (zone-y charging_station) 0.0)")
        lines.append("    (= (zone-z charging_station) 0.0)")
        lines.append("    (= (distance-to-zone drone1 charging_station) 0.0)")
        lines.append("  )")
        lines.append("")
        
        # Goal
        goal = vlm_output.get("goal", "()")
        lines.append("  (:goal")
        lines.append("    (and")
        lines.append(f"      {goal}")
        lines.append("      (landed drone1)")
        lines.append("    )")
        lines.append("  )")
        lines.append("")
        lines.append("  (:metric minimize (total-time))")
        lines.append(")")
        
        return "\n".join(lines)
    
    def update_plansys2(self, vlm_output: Dict):
        """
        Update PlanSys2 Knowledge Base via services.
        
        PlanSys2 provides these services:
        - /problem_expert/add_problem_instance
        - /problem_expert/add_problem_function
        - /problem_expert/add_problem_goal
        
        Section 6.2 of design document.
        """
        try:
            # Check if PlanSys2 is available
            from plansys2_msgs.srv import (
                AddProblemInstance,
                AddProblemFunction,
                AddProblemGoal
            )
            
            # Create service clients
            add_instance_client = self.create_client(
                AddProblemInstance,
                '/problem_expert/add_problem_instance'
            )
            
            add_function_client = self.create_client(
                AddProblemFunction,
                '/problem_expert/add_problem_function'
            )
            
            # Wait for services
            if not add_instance_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('PlanSys2 not available - skipping KB update')
                return
            
            # Add instances
            for obj in vlm_output.get("objects", []):
                req = AddProblemInstance.Request()
                req.instance = f"{obj['id']} - target"
                add_instance_client.call_async(req)
                
                # Add coordinate functions
                coords = obj.get("estimated_coords", [0, 0, 0])
                
                for axis, value in [("tx", coords[0]), ("ty", coords[1]), ("tz", coords[2])]:
                    req = AddProblemFunction.Request()
                    req.function = f"(= ({axis} {obj['id']}) {value})"
                    add_function_client.call_async(req)
            
            self.get_logger().info('PlanSys2 Knowledge Base updated')
            
        except ImportError:
            self.get_logger().debug('plansys2_msgs not available')
        except Exception as e:
            self.get_logger().warn(f'PlanSys2 update failed: {e}')
    
    def publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = VLMPDDLNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
