"""
PDDL Problem Generator for Simplified Warehouse Drone Domain

Generates PDDL problems from VLM JSON output for the simplified domain
with only 4 actions: takeoff, fly_to_target, scan_target, land.
"""

import json
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
from pathlib import Path


@dataclass
class Target:
    """Represents a target object detected by VLM"""
    id: str
    x: float
    y: float
    z: float


@dataclass
class DroneState:
    """Initial drone state"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed: float = 2.0


@dataclass
class PDDLProblem:
    """Complete PDDL problem specification"""
    problem_name: str
    drone_id: str
    targets: List[Target]
    drone_state: DroneState
    goal: str
    scan_range: float = 2.0
    takeoff_alt: float = 1.5


class PDDLGenerator:
    """Generates PDDL problem files from VLM JSON"""
    
    def __init__(self, domain_name: str = "warehouse-drone"):
        self.domain_name = domain_name
    
    def parse_vlm_json(self, vlm_data: Dict) -> PDDLProblem:
        """
        Parse VLM JSON output into PDDL problem structure
        
        Expected JSON format:
        {
            "objects": [
                {
                    "id": "blue_box_001",
                    "type": "target",
                    "estimated_coords": [2.4, -0.71, 9.36]
                }
            ],
            "goal": "(scanned blue_box_001)"
        }
        """
        # Extract targets
        targets = []
        for obj in vlm_data.get("objects", []):
            if obj.get("type") == "target" and "estimated_coords" in obj:
                coords = obj["estimated_coords"]
                targets.append(Target(
                    id=obj["id"],
                    x=coords[0],
                    y=coords[1],
                    z=coords[2]
                ))
        
        # Create problem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        problem = PDDLProblem(
            problem_name=f"vlm_generated_{timestamp}",
            drone_id="drone1",
            targets=targets,
            drone_state=DroneState(),
            goal=vlm_data.get("goal", "")
        )
        
        return problem
    
    def generate_pddl(self, problem: PDDLProblem) -> str:
        """Generate PDDL problem file content"""
        lines = []
        
        # Header
        lines.append(f";; Auto-generated PDDL Problem")
        lines.append(f";; Generated: {datetime.now().isoformat()}")
        lines.append(f";; Targets detected: {len(problem.targets)}")
        lines.append("")
        
        # Problem definition
        lines.append(f"(define (problem {problem.problem_name})")
        lines.append(f"  (:domain {self.domain_name})")
        lines.append("")
        
        # Objects
        lines.append("  (:objects")
        lines.append(f"    {problem.drone_id} - drone")
        for target in problem.targets:
            lines.append(f"    {target.id} - target")
        lines.append("  )")
        lines.append("")
        
        # Initial state
        lines.append("  (:init")
        lines.append(f"    (landed {problem.drone_id})")
        lines.append("")
        
        # Drone initial position
        lines.append(f"    ;; Drone initial position")
        lines.append(f"    (= (x {problem.drone_id}) {problem.drone_state.x})")
        lines.append(f"    (= (y {problem.drone_id}) {problem.drone_state.y})")
        lines.append(f"    (= (z {problem.drone_id}) {problem.drone_state.z})")
        lines.append("")
        
        # Targets
        for target in problem.targets:
            lines.append(f"    ;; Target: {target.id}")
            lines.append(f"    (= (tx {target.id}) {target.x})")
            lines.append(f"    (= (ty {target.id}) {target.y})")
            lines.append(f"    (= (tz {target.id}) {target.z})")
            lines.append("")
        
        lines.append("  )")
        lines.append("")
        
        # Goal
        lines.append("  (:goal")
        lines.append("    (and")
        lines.append(f"      {problem.goal}")
        lines.append(f"      (landed {problem.drone_id})")
        lines.append("    )")
        lines.append("  )")
        lines.append("")
        
        # Metric
        lines.append("  (:metric minimize (total-time))")
        lines.append(")")
        
        return "\n".join(lines)
    
    def generate_from_vlm_json(self, vlm_data: Dict) -> str:
        """
        Generate PDDL problem from VLM JSON data
        
        Args:
            vlm_data: VLM JSON output with objects and goal
            
        Returns:
            PDDL problem file content as string
        """
        problem = self.parse_vlm_json(vlm_data)
        return self.generate_pddl(problem)


if __name__ == "__main__":
    print("This module is intended to be used as a library.")
    print("Import PDDLGenerator and use generate_from_vlm_json()")
