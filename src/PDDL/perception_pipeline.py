#!/usr/bin/env python3
"""
=============================================================================
Perception Pipeline: Pixels to PDDL Coordinates
=============================================================================
Implements the depth estimation and coordinate transformation pipeline
as specified in Section 5 of the design document.

Pipeline:
1. VLM outputs bounding box (u, v, w, h) of detected object
2. Monocular depth estimation generates depth map D
3. Median depth sampling within bounding box → d_obj
4. Pinhole camera model: 2D pixels + depth → 3D camera frame
5. Pose transformation: Camera frame → World frame
=============================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import json


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters (Pinhole Model)
    
    Typically obtained from camera calibration or manufacturer specs.
    For simulation (Gazebo), check camera URDF/SDF plugin configuration.
    """
    fx: float  # Focal length X (pixels)
    fy: float  # Focal length Y (pixels)
    cx: float  # Principal point X (pixels) - usually image_width / 2
    cy: float  # Principal point Y (pixels) - usually image_height / 2
    width: int  # Image width (pixels)
    height: int  # Image height (pixels)
    
    @classmethod
    def from_fov(cls, fov_horizontal: float, width: int, height: int) -> 'CameraIntrinsics':
        """
        Create intrinsics from field of view.
        
        Args:
            fov_horizontal: Horizontal FOV in radians
            width: Image width in pixels
            height: Image height in pixels
        """
        fx = width / (2.0 * np.tan(fov_horizontal / 2.0))
        fy = fx  # Assuming square pixels
        return cls(
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height
        )
    
    @classmethod
    def default_drone_camera(cls) -> 'CameraIntrinsics':
        """
        Default intrinsics for typical drone camera (e.g., sjtu_drone in Gazebo)
        640x480 resolution, 80° horizontal FOV
        """
        return cls.from_fov(
            fov_horizontal=np.radians(80),
            width=640,
            height=480
        )


@dataclass
class DronePose:
    """
    Drone pose in world frame (from Visual Odometry or SLAM)
    """
    x: float  # World X position (meters)
    y: float  # World Y position (meters)
    z: float  # World Z position (meters)
    roll: float   # Roll (radians)
    pitch: float  # Pitch (radians)
    yaw: float    # Yaw (radians)
    
    def rotation_matrix(self) -> np.ndarray:
        """
        Compute 3x3 rotation matrix from Euler angles (ZYX convention)
        """
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R
    
    def translation_vector(self) -> np.ndarray:
        """Get translation as 3x1 vector"""
        return np.array([[self.x], [self.y], [self.z]])


class DepthEstimator:
    """
    Monocular Depth Estimation wrapper.
    
    In production, use MiDaS or Depth Anything:
    - MiDaS: https://github.com/isl-org/MiDaS
    - Depth Anything: https://github.com/LiheYoung/Depth-Anything
    
    This class provides a simple interface; the actual model loading
    is done in the subclass or via external library.
    """
    
    def __init__(self, model_type: str = "midas_v21_small"):
        """
        Initialize depth estimator.
        
        Args:
            model_type: 'midas_v21_small', 'midas_v21', 'dpt_large', 'depth_anything_v2'
        """
        self.model_type = model_type
        self.model = None
        self.transform = None
        
    def load_model(self):
        """
        Load depth estimation model.
        
        For actual implementation, uncomment and use:
        
        import torch
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ["dpt_large", "dpt_hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        """
        print(f"[DepthEstimator] Would load model: {self.model_type}")
        
    def estimate_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Depth map as numpy array (H, W) in relative depth units
            
        Note:
            MiDaS outputs RELATIVE depth. For metric depth, you need
            additional scale calibration or use Depth Anything V2 with
            metric fine-tuning.
        """
        # Placeholder - returns simulated depth map
        # In production, run actual model inference
        H, W = rgb_image.shape[:2]
        
        # Simulate depth with gradient (closer = larger values for MiDaS)
        simulated_depth = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            # Objects at center-bottom are "closer"
            simulated_depth[y, :] = 10.0 - (y / H) * 5.0  # 5-10m range
            
        return simulated_depth
    
    def sample_depth_in_bbox(
        self, 
        depth_map: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Sample depth within bounding box using median.
        
        As per design document Section 5.1:
        d_obj = median(D[v:v+h, u:u+w])
        
        Args:
            depth_map: Full depth map (H, W)
            bbox: Bounding box (u, v, w, h) in pixels
            
        Returns:
            Median depth value within bbox
        """
        u, v, w, h = bbox
        
        # Clamp to image bounds
        u = max(0, u)
        v = max(0, v)
        u_end = min(depth_map.shape[1], u + w)
        v_end = min(depth_map.shape[0], v + h)
        
        # Extract region and compute median
        region = depth_map[v:v_end, u:u_end]
        
        if region.size == 0:
            return 5.0  # Default depth
            
        return float(np.median(region))


class CoordinateTransformer:
    """
    Transforms 2D image coordinates + depth to 3D world coordinates.
    
    Implements Section 5.2 of the design document:
    1. Pinhole camera model: (u, v, d) → (X_cam, Y_cam, Z_cam)
    2. Pose transformation: P_cam → P_world
    """
    
    def __init__(self, camera_intrinsics: CameraIntrinsics):
        self.K = camera_intrinsics
        
    def pixel_to_camera_frame(
        self, 
        u: float, 
        v: float, 
        depth: float
    ) -> np.ndarray:
        """
        Transform pixel coordinates + depth to camera 3D frame.
        
        Pinhole Camera Model (from design document):
            X_cam = (u_c - c_x) * d_obj / f_x
            Y_cam = (v_c - c_y) * d_obj / f_y
            Z_cam = d_obj
            
        Args:
            u: Pixel X coordinate (center of bbox)
            v: Pixel Y coordinate (center of bbox)
            depth: Estimated depth (meters)
            
        Returns:
            3D point in camera frame as (3,1) numpy array
        """
        X_cam = (u - self.K.cx) * depth / self.K.fx
        Y_cam = (v - self.K.cy) * depth / self.K.fy
        Z_cam = depth
        
        return np.array([[X_cam], [Y_cam], [Z_cam]])
    
    def camera_to_world_frame(
        self,
        P_cam: np.ndarray,
        drone_pose: DronePose
    ) -> np.ndarray:
        """
        Transform point from camera frame to world frame.
        
        P_world = R_drone * P_cam + T_drone
        
        Note: This assumes camera is aligned with drone body frame.
        For camera-to-body offset, add additional transformation.
        
        Args:
            P_cam: Point in camera frame (3,1)
            drone_pose: Current drone pose
            
        Returns:
            Point in world frame (3,1)
        """
        R = drone_pose.rotation_matrix()
        T = drone_pose.translation_vector()
        
        P_world = R @ P_cam + T
        
        return P_world
    
    def pixel_to_world(
        self,
        u: float,
        v: float,
        depth: float,
        drone_pose: DronePose
    ) -> Tuple[float, float, float]:
        """
        Full pipeline: pixel + depth → world coordinates.
        
        Args:
            u, v: Pixel coordinates (bbox center)
            depth: Estimated depth (meters)
            drone_pose: Current drone pose
            
        Returns:
            (x, y, z) in world frame (meters)
        """
        # Step 1: Pixel → Camera frame
        P_cam = self.pixel_to_camera_frame(u, v, depth)
        
        # Step 2: Camera → World frame
        P_world = self.camera_to_world_frame(P_cam, drone_pose)
        
        return (
            float(P_world[0, 0]),
            float(P_world[1, 0]),
            float(P_world[2, 0])
        )


class PerceptionPipeline:
    """
    Complete perception pipeline: Image + VLM → PDDL-ready JSON.
    
    Integrates:
    - VLM output parsing
    - Depth estimation
    - Coordinate transformation
    - JSON generation for PDDL converter
    """
    
    def __init__(
        self,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        depth_model: str = "midas_v21_small"
    ):
        self.camera = camera_intrinsics or CameraIntrinsics.default_drone_camera()
        self.depth_estimator = DepthEstimator(depth_model)
        self.coord_transformer = CoordinateTransformer(self.camera)
        
    def process_vlm_output(
        self,
        rgb_image: np.ndarray,
        vlm_detections: List[Dict],
        drone_pose: DronePose,
        instruction: str
    ) -> Dict:
        """
        Process VLM detections and generate PDDL-ready JSON.
        
        Args:
            rgb_image: RGB image from drone camera
            vlm_detections: List of VLM detections, each with:
                - id: Object identifier
                - bbox: [u, v, w, h] bounding box
            drone_pose: Current drone pose
            instruction: Natural language instruction
            
        Returns:
            JSON dict ready for PDDL generator
        """
        # Get depth map
        depth_map = self.depth_estimator.estimate_depth(rgb_image)
        
        objects = []
        
        for detection in vlm_detections:
            obj_id = detection["id"]
            bbox = detection["bbox"]  # [u, v, w, h]
            
            # Calculate bbox center
            u_center = bbox[0] + bbox[2] / 2
            v_center = bbox[1] + bbox[3] / 2
            
            # Sample depth within bbox
            depth = self.depth_estimator.sample_depth_in_bbox(depth_map, tuple(bbox))
            
            # Transform to world coordinates
            x, y, z = self.coord_transformer.pixel_to_world(
                u_center, v_center, depth, drone_pose
            )
            
            objects.append({
                "id": obj_id,
                "type": "target",
                "bbox": bbox,
                "estimated_coords": [round(x, 2), round(y, 2), round(z, 2)]
            })
        
        # Infer goal from instruction
        goal = self._infer_goal_from_instruction(instruction, objects)
        
        return {
            "instruction": instruction,
            "objects": objects,
            "goal": goal
        }
    
    def _infer_goal_from_instruction(
        self,
        instruction: str,
        objects: List[Dict]
    ) -> str:
        """
        Simple rule-based goal inference from instruction.
        
        In production, this would be part of VLM output.
        """
        instruction_lower = instruction.lower()
        
        if not objects:
            return "()" 
            
        obj_id = objects[0]["id"]
        
        if "scan" in instruction_lower or "inspect" in instruction_lower:
            return f"(scanned {obj_id})"
        elif "go to" in instruction_lower or "fly to" in instruction_lower:
            # For navigation-only goals, we'd need additional predicates
            return f"(scanned {obj_id})"  # Default to scan
        else:
            return f"(scanned {obj_id})"


def demo():
    """Demonstrate the perception pipeline"""
    
    print("=" * 70)
    print("Perception Pipeline Demo: Pixels to PDDL Coordinates")
    print("=" * 70)
    
    # Simulated inputs
    print("\n1. Camera Intrinsics (Drone Camera)")
    camera = CameraIntrinsics.default_drone_camera()
    print(f"   fx={camera.fx:.1f}, fy={camera.fy:.1f}")
    print(f"   cx={camera.cx:.1f}, cy={camera.cy:.1f}")
    print(f"   Resolution: {camera.width}x{camera.height}")
    
    print("\n2. Drone Pose (from SLAM/VO)")
    drone_pose = DronePose(
        x=0.0, y=0.0, z=1.5,
        roll=0.0, pitch=0.0, yaw=0.0
    )
    print(f"   Position: ({drone_pose.x}, {drone_pose.y}, {drone_pose.z}) m")
    print(f"   Orientation: roll={drone_pose.roll}, pitch={drone_pose.pitch}, yaw={drone_pose.yaw} rad")
    
    print("\n3. VLM Detection (PaliGemma output)")
    vlm_detection = {
        "id": "red_box_01",
        "bbox": [320, 240, 100, 80]  # Center of image
    }
    print(f"   ID: {vlm_detection['id']}")
    print(f"   BBox: {vlm_detection['bbox']}")
    
    print("\n4. Depth Estimation")
    # Simulated RGB image
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate_depth(rgb_image)
    depth = depth_estimator.sample_depth_in_bbox(
        depth_map, 
        tuple(vlm_detection['bbox'])
    )
    print(f"   Sampled depth: {depth:.2f} m")
    
    print("\n5. Coordinate Transformation (Pinhole Model)")
    coord_transformer = CoordinateTransformer(camera)
    
    # BBox center
    u_c = vlm_detection['bbox'][0] + vlm_detection['bbox'][2] / 2
    v_c = vlm_detection['bbox'][1] + vlm_detection['bbox'][3] / 2
    print(f"   BBox center: ({u_c}, {v_c}) pixels")
    
    # Camera frame
    P_cam = coord_transformer.pixel_to_camera_frame(u_c, v_c, depth)
    print(f"   Camera frame: ({P_cam[0,0]:.2f}, {P_cam[1,0]:.2f}, {P_cam[2,0]:.2f}) m")
    
    # World frame
    x, y, z = coord_transformer.pixel_to_world(u_c, v_c, depth, drone_pose)
    print(f"   World frame: ({x:.2f}, {y:.2f}, {z:.2f}) m")
    
    print("\n6. PDDL-Ready Output")
    pddl_output = {
        "objects": [{
            "id": vlm_detection['id'],
            "type": "target",
            "bbox": vlm_detection['bbox'],
            "estimated_coords": [round(x, 2), round(y, 2), round(z, 2)]
        }],
        "goal": f"(scanned {vlm_detection['id']})"
    }
    print(json.dumps(pddl_output, indent=2))
    
    print("\n7. PDDL Fluent Values")
    print(f"   (= (tx {vlm_detection['id']}) {x:.2f})")
    print(f"   (= (ty {vlm_detection['id']}) {y:.2f})")
    print(f"   (= (tz {vlm_detection['id']}) {z:.2f})")
    
    print("\n" + "=" * 70)
    print("✓ Pipeline completed successfully")


if __name__ == "__main__":
    demo()
