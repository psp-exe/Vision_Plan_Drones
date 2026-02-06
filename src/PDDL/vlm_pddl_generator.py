#!/usr/bin/env python3
"""
=============================================================================
VLM PDDL Problem Generator
=============================================================================
Converts Vision Language Model JSON output to valid PDDL problem files.

This script implements the "Intermediate Representation" strategy from the
design document: VLM outputs structured JSON → This script converts to PDDL

Pipeline:
1. VLM (PaliGemma/LLaVA) detects objects and outputs JSON schema
2. Depth estimation provides Z coordinate
3. Pinhole camera model converts 2D→3D coordinates
4. This script generates the PDDL problem file
=============================================================================
"""

import json
import math
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Target:
    """Represents a detected target object"""
    id: str
    type: str
    x: float
    y: float
    z: float
    bbox: Optional[List[int]] = None  # [u, v, w, h] in pixels


@dataclass
class DroneState:
    """Current drone state"""
    id: str = "drone1"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    battery_level: float = 95.0
    battery_capacity: float = 100.0
    fly_speed: float = 2.0
    discharge_rate_fly: float = 0.5
    discharge_rate_hover: float = 0.2
    recharge_rate: float = 1.0
    is_landed: bool = True
    is_calibrated: bool = True
    is_docked: bool = True


@dataclass
class Zone:
    """Static zone definition"""
    id: str
    x: float
    y: float
    z: float


@dataclass
class PDDLProblem:
    """Complete PDDL problem configuration"""
    name: str
    domain: str = "warehouse-drone"
    drone: DroneState = field(default_factory=DroneState)
    targets: List[Target] = field(default_factory=list)
    zones: List[Zone] = field(default_factory=list)
    goal: str = ""
    scan_range: float = 2.0
    min_battery: float = 20.0
    takeoff_altitude: float = 1.5


class VLMOutputSchema:
    """
    JSON Schema that the VLM is fine-tuned to output.
    Used for grammar-constrained decoding during inference.
    """
    SCHEMA = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string", "enum": ["target"]},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "estimated_coords": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "required": ["id", "type", "bbox", "estimated_coords"]
                }
            },
            "goal": {"type": "string"},
            "instruction": {"type": "string"}
        },
        "required": ["objects", "goal"]
    }


class PDDLGenerator:
    """
    Generates PDDL problem files from VLM output.
    
    Key responsibilities:
    1. Parse VLM JSON output
    2. Calculate Euclidean distances
    3. Generate syntactically valid PDDL
    """
    
    def __init__(self, domain_name: str = "warehouse-drone"):
        self.domain_name = domain_name
        
    def euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        """Calculate 3D Euclidean distance"""
        return math.sqrt(
            (p1[0] - p2[0])**2 + 
            (p1[1] - p2[1])**2 + 
            (p1[2] - p2[2])**2
        )
    
    def parse_vlm_json(self, vlm_output: Dict) -> PDDLProblem:
        """
        Parse VLM JSON output into PDDLProblem dataclass.
        
        Expected VLM output format:
        {
            "objects": [
                {
                    "id": "red_box_01",
                    "type": "target",
                    "bbox": [320, 240, 100, 80], 
                    "estimated_coords": [12.5, 3.2, 4.0]
                }
            ],
            "goal": "(scanned red_box_01)",
            "instruction": "Scan the red box"
        }
        """
        problem = PDDLProblem(
            name=f"vlm_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Parse detected objects
        for obj in vlm_output.get("objects", []):
            coords = obj.get("estimated_coords", [0, 0, 0])
            target = Target(
                id=obj["id"],
                type=obj.get("type", "target"),
                x=coords[0],
                y=coords[1],
                z=coords[2],
                bbox=obj.get("bbox")
            )
            problem.targets.append(target)
        
        # Parse goal
        problem.goal = vlm_output.get("goal", "")
        
        # Add default charging station
        problem.zones.append(Zone(id="charging_station", x=0.0, y=0.0, z=0.0))
        
        return problem
    
    def generate_pddl(self, problem: PDDLProblem) -> str:
        """
        Generate complete PDDL problem file.
        
        Returns:
            Valid PDDL 2.1 problem file as string
        """
        lines = []
        
        # Header
        lines.append(f";; Auto-generated PDDL Problem")
        lines.append(f";; Generated: {datetime.now().isoformat()}")
        lines.append(f";; Targets detected: {len(problem.targets)}")
        lines.append("")
        
        # Problem definition
        lines.append(f"(define (problem {problem.name})")
        lines.append(f"  (:domain {problem.domain})")
        lines.append("")
        
        # Objects
        lines.append("  (:objects")
        lines.append(f"    {problem.drone.id} - drone")
        for target in problem.targets:
            lines.append(f"    {target.id} - target")
        for zone in problem.zones:
            lines.append(f"    {zone.id} - zone")
        lines.append("  )")
        lines.append("")
        
        # Initial state
        lines.append("  (:init")
        
        # Drone predicates
        if problem.drone.is_landed:
            lines.append(f"    (landed {problem.drone.id})")
        else:
            lines.append(f"    (airborne {problem.drone.id})")
        if problem.drone.is_calibrated:
            lines.append(f"    (calibrated {problem.drone.id})")
        if problem.drone.is_docked:
            lines.append(f"    (docked {problem.drone.id})")
            lines.append(f"    (at-zone {problem.drone.id} charging_station)")
        lines.append("")
        
        # Drone position
        lines.append(f"    ;; Drone position")
        lines.append(f"    (= (x {problem.drone.id}) {problem.drone.x})")
        lines.append(f"    (= (y {problem.drone.id}) {problem.drone.y})")
        lines.append(f"    (= (z {problem.drone.id}) {problem.drone.z})")
        lines.append("")
        
        # Battery parameters
        lines.append(f"    ;; Battery parameters")
        lines.append(f"    (= (battery-level {problem.drone.id}) {problem.drone.battery_level})")
        lines.append(f"    (= (battery-capacity {problem.drone.id}) {problem.drone.battery_capacity})")
        lines.append(f"    (= (discharge-rate-fly {problem.drone.id}) {problem.drone.discharge_rate_fly})")
        lines.append(f"    (= (discharge-rate-hover {problem.drone.id}) {problem.drone.discharge_rate_hover})")
        lines.append(f"    (= (recharge-rate {problem.drone.id}) {problem.drone.recharge_rate})")
        lines.append("")
        
        # Kinematics
        lines.append(f"    ;; Kinematics")
        lines.append(f"    (= (fly-speed {problem.drone.id}) {problem.drone.fly_speed})")
        lines.append("")
        
        # Operational thresholds
        lines.append(f"    ;; Operational thresholds")
        lines.append(f"    (= (scan-range) {problem.scan_range})")
        lines.append(f"    (= (min-battery) {problem.min_battery})")
        lines.append(f"    (= (takeoff-altitude) {problem.takeoff_altitude})")
        lines.append("")
        
        # Target coordinates (VLM output)
        for target in problem.targets:
            lines.append(f"    ;; Target: {target.id} (VLM detected)")
            lines.append(f"    (= (tx {target.id}) {target.x})")
            lines.append(f"    (= (ty {target.id}) {target.y})")
            lines.append(f"    (= (tz {target.id}) {target.z})")
            
            # Calculate distance from drone
            drone_pos = (problem.drone.x, problem.drone.y, problem.drone.z)
            target_pos = (target.x, target.y, target.z)
            distance = round(self.euclidean_distance(drone_pos, target_pos), 2)
            lines.append(f"    (= (distance-to {problem.drone.id} {target.id}) {distance})")
            lines.append("")
        
        # Zone coordinates
        for zone in problem.zones:
            lines.append(f"    ;; Zone: {zone.id}")
            lines.append(f"    (= (zone-x {zone.id}) {zone.x})")
            lines.append(f"    (= (zone-y {zone.id}) {zone.y})")
            lines.append(f"    (= (zone-z {zone.id}) {zone.z})")
            
            # Distance to zone
            drone_pos = (problem.drone.x, problem.drone.y, problem.drone.z)
            zone_pos = (zone.x, zone.y, zone.z)
            distance = round(self.euclidean_distance(drone_pos, zone_pos), 2)
            lines.append(f"    (= (distance-to-zone {problem.drone.id} {zone.id}) {distance})")
        
        lines.append("  )")
        lines.append("")
        
        # Goal
        lines.append("  (:goal")
        lines.append("    (and")
        lines.append(f"      {problem.goal}")
        lines.append(f"      (landed {problem.drone.id})")
        lines.append("    )")
        lines.append("  )")
        lines.append("")
        
        # Metric (minimize time)
        lines.append("  (:metric minimize (total-time))")
        lines.append(")")
        
        return "\n".join(lines)
    
    def generate_from_vlm_json(self, vlm_json: Dict) -> str:
        """
        End-to-end: VLM JSON → PDDL problem file
        
        Args:
            vlm_json: JSON output from fine-tuned VLM
            
        Returns:
            Valid PDDL problem file string
        """
        problem = self.parse_vlm_json(vlm_json)
        return self.generate_pddl(problem)


def main():
    """Demonstrate VLM JSON → PDDL conversion"""
    
    # Example VLM output (as if from PaliGemma/LLaVA inference)
    vlm_output = {
        "instruction": "Scan the red hazardous container",
        "objects": [
            {
                "id": "haz_container_01",
                "type": "target",
                "bbox": [320, 240, 100, 80],
                "estimated_coords": [10.5, 2.0, 3.5]
            }
        ],
        "goal": "(scanned haz_container_01)"
    }
    
    print("=" * 70)
    print("VLM PDDL Problem Generator")
    print("=" * 70)
    print("\nInput (VLM JSON output):")
    print(json.dumps(vlm_output, indent=2))
    print("\n" + "=" * 70)
    
    generator = PDDLGenerator()
    pddl_content = generator.generate_from_vlm_json(vlm_output)
    
    print("\nGenerated PDDL Problem:")
    print("-" * 70)
    print(pddl_content)
    
    # Save to file
    output_path = "/home/psp/ros_ws/src/PDDL/generated_problem.pddl"
    with open(output_path, 'w') as f:
        f.write(pddl_content)
    print("\n" + "=" * 70)
    print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
