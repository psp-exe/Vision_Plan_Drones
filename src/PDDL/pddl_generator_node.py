#!/usr/bin/env python3
"""
ROS Node: PDDL Problem Generator

Reads from flight_plan.json and generates PDDL problem files
that adhere to warehouse_drone_domain.pddl
"""

import rclpy
from rclpy.node import Node
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# Import the PDDL generator library
from vlm_pddl_generator import PDDLGenerator


class PDDLGeneratorNode(Node):
    """
    ROS 2 Node that generates PDDL problem files from flight plan JSON
    """
    
    def __init__(self):
        super().__init__('pddl_generator_node')
        
        # Initialize PDDL generator
        self.generator = PDDLGenerator(domain_name="warehouse-drone")
        
        # File paths
        self.flight_plan_path = "/home/psp/ros_ws/src/vpdrones/src/sjtu_drone/vpdrones/flight_plan.json"
        self.output_dir = "/home/psp/ros_ws/src/PDDL/generated_problems"
        self.domain_file = "/home/psp/ros_ws/src/PDDL/warehouse_drone_domain.pddl"
        self.solver_script = "/home/psp/ros_ws/src/PDDL/pddl_solver.py"
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info("PDDL Generator Node initialized")
        self.get_logger().info(f"Reading flight plan from: {self.flight_plan_path}")
        self.get_logger().info(f"Saving PDDL problems to: {self.output_dir}")
        self.get_logger().info("Press 'Enter' to generate PDDL problem and solve.")
        
        # Start input thread
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()
    
    def read_flight_plan(self) -> dict:
        """
        Read and parse the flight plan JSON file
        
        Returns:
            Flight plan data as dictionary
        """
        flight_plan_file = Path(self.flight_plan_path)
        
        if not flight_plan_file.exists():
            raise FileNotFoundError(f"Flight plan not found: {self.flight_plan_path}")
        
        with open(flight_plan_file, 'r') as f:
            flight_plan = json.load(f)
        
        return flight_plan
    
    def wait_for_input(self):
        """Wait for user to press Enter, then generate and solve."""
        while rclpy.ok():
            try:
                input()
                self.get_logger().info("Generating PDDL problem...")
                problem_path = self.generate_pddl()
                
                if problem_path:
                    self.run_solver(problem_path)
                
                self.get_logger().info("Press 'Enter' to generate again.")
            except EOFError:
                break
    
    def run_solver(self, problem_path):
        """Run pddl_solver.py on the generated problem file."""
        self.get_logger().info("Running PDDL solver...")
        try:
            result = subprocess.run(
                ['python3', self.solver_script],
                capture_output=True,
                text=True,
                cwd='/home/psp/ros_ws'
            )
            # Print solver output
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0 and result.stderr:
                self.get_logger().error(f"Solver error: {result.stderr}")
        except Exception as e:
            self.get_logger().error(f"Failed to run solver: {e}")

    def generate_pddl(self):
        """Generate PDDL problem from flight plan. Returns path to generated file or None."""
        try:
            flight_plan_file = Path(self.flight_plan_path)
            
            if not flight_plan_file.exists():
                self.get_logger().warn(f"Flight plan file not found: {self.flight_plan_path}")
                return None
            
            # Read flight plan
            flight_plan = self.read_flight_plan()
            
            # Validate that objects have estimated_coords
            objects = flight_plan.get("objects", [])
            valid_objects = []
            for obj in objects:
                if "estimated_coords" not in obj:
                    self.get_logger().warn(
                        f"Object {obj.get('id', 'unknown')} missing estimated_coords, skipping"
                    )
                    continue
                valid_objects.append(obj)
            
            if not valid_objects:
                self.get_logger().warn("No valid objects with coordinates in flight plan")
                return None
            
            # Update flight plan with only valid objects
            flight_plan["objects"] = valid_objects
            
            # Generate PDDL problem
            pddl_content = self.generator.generate_from_vlm_json(flight_plan)
            
            # Save to file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"problem_{timestamp}.pddl"
            output_path = Path(self.output_dir) / output_filename
            
            with open(output_path, 'w') as f:
                f.write(pddl_content)
            
            # Also save as latest problem
            latest_path = Path(self.output_dir) / "problem_latest.pddl"
            with open(latest_path, 'w') as f:
                f.write(pddl_content)
            
            self.get_logger().info(f"✓ Generated PDDL problem: {output_filename}")
            self.get_logger().info(f"  - Objects: {len(valid_objects)}")
            self.get_logger().info(f"  - Goal: {flight_plan.get('goal', 'N/A')}")
            
            for obj in valid_objects:
                coords = obj.get('estimated_coords', [0, 0, 0])
                self.get_logger().info(
                    f"    • {obj['id']}: ({coords[0]}, {coords[1]}, {coords[2]}) m"
                )
            
            return latest_path
        
        except FileNotFoundError as e:
            self.get_logger().error(f"File not found: {e}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in flight plan: {e}")
        except Exception as e:
            self.get_logger().error(f"Failed to generate PDDL: {e}")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = PDDLGeneratorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
