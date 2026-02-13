#!/usr/bin/env python3
"""
PDDL Solver Script using PlanSys2

Solves PDDL planning problems using PlanSys2's POPF planner
and converts the plan output to JSON format.
"""

import subprocess
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PlanSys2Solver:
    """
    PDDL solver using PlanSys2/POPF planner
    """
    
    def __init__(self, popf_path: str = "/opt/ros/humble/lib/popf/popf"):
        self.popf_path = popf_path
        
    def solve(self, domain_file: str, problem_file: str, output_dir: str = None) -> Dict:
        """
        Solve PDDL problem and convert to JSON
        
        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            output_dir: Directory to save plan JSON (optional)
            
        Returns:
            Dictionary containing plan information
        """
        print(f"Solving PDDL problem...")
        print(f"  Domain: {domain_file}")
        print(f"  Problem: {problem_file}")
        
        # Run POPF planner
        try:
            result = subprocess.run(
                [self.popf_path, domain_file, problem_file],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                print(f"Planner error (exit code {result.returncode}):")
                print(result.stderr)
                return self._create_error_response(result.stderr)
            
            # Parse POPF output
            plan_data = self._parse_popf_output(result.stdout, problem_file)
            
            # Save to JSON if output directory specified
            if output_dir:
                self._save_plan_json(plan_data, output_dir)
            
            return plan_data
            
        except subprocess.TimeoutExpired:
            print("Planner timed out after 60 seconds")
            return self._create_error_response("Timeout: Planning took longer than 60 seconds")
        except FileNotFoundError:
            print(f"Error: POPF binary not found at {self.popf_path}")
            return self._create_error_response(f"POPF not found at {self.popf_path}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._create_error_response(str(e))
    
    def _parse_target_coords(self, problem_file: str) -> Dict[str, Dict[str, float]]:
        """
        Parse target 3D coordinates from PDDL problem file.
        
        Looks for lines like:
            (= (tx target_name) 2.4)
            (= (ty target_name) -0.71)
            (= (tz target_name) 9.36)
        
        Returns:
            Dict mapping target name -> {"x": ..., "y": ..., "z": ...}
        """
        coords = {}  # target_name -> {"x": ..., "y": ..., "z": ...}
        
        try:
            with open(problem_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return coords
        
        # Match (= (tx target_name) value), (= (ty ...) ...), (= (tz ...) ...)
        for axis, key in [('tx', 'x'), ('ty', 'y'), ('tz', 'z')]:
            pattern = rf'\(=\s*\({axis}\s+([^)]+)\)\s+([\-\d.]+)\)'
            for match in re.finditer(pattern, content):
                target_name = match.group(1).strip()
                value = float(match.group(2))
                if target_name not in coords:
                    coords[target_name] = {}
                coords[target_name][key] = value
        
        return coords

    def _parse_popf_output(self, output: str, problem_file: str) -> Dict:
        """
        Parse POPF planner output into structured JSON
        
        POPF output format:
        0.000: (action param1 param2) [duration]
        """
        lines = output.strip().split('\n')
        
        # Extract target coordinates from problem file
        target_coords = self._parse_target_coords(problem_file)
        
        actions = []
        plan_found = False
        makespan = 0.0
        
        for line in lines:
            line = line.strip()
            
            # Check if plan was found
            if "Solution Found" in line or "Plan found" in line:
                plan_found = True
            
            # Parse action lines: "0.000: (action param1 param2) [5.000]"
            action_match = re.match(r'([\d.]+):\s*\(([^)]+)\)\s*\[([\d.]+)\]', line)
            if action_match:
                start_time = float(action_match.group(1))
                action_str = action_match.group(2).strip()
                duration = float(action_match.group(3))
                
                # Parse action name and parameters
                parts = action_str.split()
                action_name = parts[0] if parts else ""
                parameters = parts[1:] if len(parts) > 1 else []
                
                end_time = start_time + duration
                makespan = max(makespan, end_time)
                
                action_entry = {
                    "name": action_name,
                    "parameters": parameters,
                    "start_time": round(start_time, 3),
                    "duration": round(duration, 3),
                    "end_time": round(end_time, 3)
                }
                
                # Enrich fly_to_target with target 3D coordinates
                if action_name == "fly_to_target" and len(parameters) >= 2:
                    target_name = parameters[1]
                    if target_name in target_coords:
                        action_entry["coordinates"] = target_coords[target_name]
                
                actions.append(action_entry)
        
        # If no actions but plan found mentioned, check for simpler format
        if not actions and plan_found:
            for line in lines:
                # Sometimes POPF uses simpler format: "(action param1 param2)"
                simple_match = re.match(r'\(([^)]+)\)', line)
                if simple_match:
                    action_str = simple_match.group(1).strip()
                    parts = action_str.split()
                    if parts:
                        actions.append({
                            "name": parts[0],
                            "parameters": parts[1:] if len(parts) > 1 else [],
                            "start_time": 0.0,
                            "duration": 0.0,
                            "end_time": 0.0
                        })
        
        problem_name = Path(problem_file).stem
        
        return {
            "problem": problem_name,
            "plan_found": plan_found or len(actions) > 0,
            "actions": actions,
            "makespan": round(makespan, 3),
            "metric": round(makespan, 3),
            "num_actions": len(actions),
            "generated_at": datetime.now().isoformat()
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response JSON"""
        return {
            "problem": "unknown",
            "plan_found": False,
            "actions": [],
            "makespan": 0.0,
            "metric": 0.0,
            "num_actions": 0,
            "error": error_message,
            "generated_at": datetime.now().isoformat()
        }
    
    def _save_plan_json(self, plan_data: Dict, output_dir: str):
        """Save plan to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save timestamped version
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_file = output_path / f"plan_{timestamp}.json"
        
        with open(timestamped_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        # Save as latest
        latest_file = output_path / "plan_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        print(f"\n✓ Plan saved to:")
        print(f"  - {timestamped_file}")
        print(f"  - {latest_file}")


def main():
    """Main function"""
    domain = '/home/psp/ros_ws/src/PDDL/warehouse_drone_domain.pddl'
    problem = '/home/psp/ros_ws/src/PDDL/generated_problems/problem_latest.pddl'
    output = '/home/psp/ros_ws/src/PDDL/generated_plans'
    
    # Solve
    solver = PlanSys2Solver()
    plan_data = solver.solve(domain, problem, output)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PLAN SUMMARY")
    print("=" * 70)
    print(f"Problem: {plan_data['problem']}")
    print(f"Plan Found: {plan_data['plan_found']}")
    print(f"Number of Actions: {plan_data['num_actions']}")
    print(f"Makespan: {plan_data['makespan']} seconds")
    
    if plan_data['plan_found'] and plan_data['actions']:
        print(f"\nActions:")
        for i, action in enumerate(plan_data['actions'], 1):
            params_str = ' '.join(action['parameters'])
            print(f"  {i}. [{action['start_time']:.2f}s] {action['name']} {params_str} "
                  f"[{action['duration']:.2f}s]")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
