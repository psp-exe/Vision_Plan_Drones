#!/usr/bin/env python3
"""
=============================================================================
Perception Pipeline: Pixels to PDDL Coordinates
=============================================================================
Implements depth estimation and coordinate transformation:

Pipeline:
1. VLM outputs bounding box (u, v, w, h) of detected object
2. Monocular depth estimation generates depth map D
3. Median depth sampling within bounding box → d_obj
4. Pinhole camera model: 2D pixels + depth → 3D camera frame
5. Camera optical frame → Body frame → World frame

Camera Conventions:
  - Camera optical frame (OpenCV): X-right, Y-down, Z-forward
  - Body frame (ROS/Gazebo):       X-forward, Y-left, Z-up
  - World frame:                    X-east, Y-north, Z-up

sjtu_drone front camera (from URDF):
  - Joint: xyz="0.2 0 0", rpy="0 0 0" (no rotation from body)
  - FOV: 2.09 rad (≈120°), Resolution: 640x360
=============================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters (Pinhole Model)
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
        fy = fx  # Square pixels
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
        Intrinsics for sjtu_drone front camera (from URDF):
        640x360 resolution, 2.09 rad (~120°) horizontal FOV
        """
        return cls.from_fov(
            fov_horizontal=2.09,  # radians, from URDF
            width=640,
            height=360
        )


@dataclass
class DronePose:
    """
    Drone pose in world frame (from /drone/gt_pose)
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
        R = Rz(yaw) * Ry(pitch) * Rx(roll)
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


# Rotation from camera optical frame to body frame.
# Camera optical: X-right, Y-down, Z-forward
# Body frame:     X-forward, Y-left, Z-up
#
# body_X = cam_Z  (forward)
# body_Y = -cam_X (left = -right)
# body_Z = -cam_Y (up = -down)
R_CAM_TO_BODY = np.array([
    [0,  0, 1],   # body_x = cam_z
    [-1, 0, 0],   # body_y = -cam_x
    [0, -1, 0]    # body_z = -cam_y
])

# Camera mount offset in body frame (from URDF: xyz="0.2 0 0")
T_CAM_IN_BODY = np.array([[0.2], [0.0], [0.0]])


class DepthEstimator:
    """
    Monocular Depth Estimation wrapper.
    
    Currently uses a simple heuristic based on bounding box size.
    In production, use MiDaS or Depth Anything for real depth maps.
    """
    
    def __init__(self, model_type: str = "midas_v21_small"):
        self.model_type = model_type
        self.model = None
        
    def load_model(self):
        """Load depth estimation model (placeholder)."""
        print(f"[DepthEstimator] Would load model: {self.model_type}")
        
    def estimate_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Returns:
            Depth map as numpy array (H, W) in meters.
            
        Note: This is currently a placeholder. For real depth, 
        integrate MiDaS or Depth Anything V2.
        """
        H, W = rgb_image.shape[:2]
        # Placeholder: uniform depth, will be overridden by bbox-based estimation
        return np.full((H, W), 5.0, dtype=np.float32)
    
    def estimate_depth_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_height: int,
        known_object_height: float = 0.3
    ) -> float:
        """
        Estimate depth from bounding box size using perspective geometry.
        
        The idea: if we know the real-world height of the object,
        depth ≈ (f_y * real_height) / bbox_pixel_height
        
        For objects of unknown size, we use a heuristic based on
        the bbox vertical position (objects lower in the image are closer).
        
        Args:
            bbox: (u, v, w, h) bounding box
            image_height: Total image height in pixels
            known_object_height: Estimated real-world height in meters
            
        Returns:
            Estimated depth in meters
        """
        _, v, _, h = bbox
        
        if h <= 0:
            return 10.0  # Default far depth
        
        # Use the sjtu_drone camera focal length
        cam = CameraIntrinsics.default_drone_camera()
        
        # Depth from similar triangles: d = (fy * H_real) / h_pixels
        depth = (cam.fy * known_object_height) / h
        
        # Clamp to reasonable range
        depth = max(0.5, min(depth, 50.0))
        
        return depth
    
    def sample_depth_in_bbox(
        self, 
        depth_map: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Sample depth within bounding box using median.
        d_obj = median(D[v:v+h, u:u+w])
        """
        u, v, w, h = bbox
        
        u = max(0, u)
        v = max(0, v)
        u_end = min(depth_map.shape[1], u + w)
        v_end = min(depth_map.shape[0], v + h)
        
        region = depth_map[v:v_end, u:u_end]
        
        if region.size == 0:
            return 5.0
            
        return float(np.median(region))


class CoordinateTransformer:
    """
    Transforms 2D image coordinates + depth to 3D world coordinates.
    
    Pipeline:
    1. Pinhole model: (u, v, d) → P_optical  (camera optical frame)
    2. Optical → Body: P_body = R_cam_to_body @ P_optical + T_cam_offset
    3. Body → World:   P_world = R_drone @ P_body + T_drone
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
        Pixel coords + depth → 3D point in camera OPTICAL frame.
        
        Camera optical frame (OpenCV convention):
            X_cam = (u - cx) * depth / fx   (right)
            Y_cam = (v - cy) * depth / fy   (down)
            Z_cam = depth                    (forward, into scene)
        """
        X_cam = (u - self.K.cx) * depth / self.K.fx
        Y_cam = (v - self.K.cy) * depth / self.K.fy
        Z_cam = depth
        
        return np.array([[X_cam], [Y_cam], [Z_cam]])
    
    def camera_to_world_frame(
        self,
        P_optical: np.ndarray,
        drone_pose: DronePose
    ) -> np.ndarray:
        """
        Transform from camera optical frame → world frame.
        
        Steps:
        1. Optical → Body:  P_body = R_cam_to_body @ P_optical + T_cam_offset
        2. Body → World:    P_world = R_drone @ P_body + T_drone
        """
        # Step 1: Camera optical → Body frame
        P_body = R_CAM_TO_BODY @ P_optical + T_CAM_IN_BODY
        
        # Step 2: Body → World
        R = drone_pose.rotation_matrix()
        T = drone_pose.translation_vector()
        P_world = R @ P_body + T
        
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
        """
        P_optical = self.pixel_to_camera_frame(u, v, depth)
        P_world = self.camera_to_world_frame(P_optical, drone_pose)
        
        return (
            float(P_world[0, 0]),
            float(P_world[1, 0]),
            float(P_world[2, 0])
        )


class PerceptionPipeline:
    """
    Complete perception pipeline: Image + VLM → PDDL-ready JSON.
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
        """
        depth_map = self.depth_estimator.estimate_depth(rgb_image)
        
        objects = []
        
        for detection in vlm_detections:
            obj_id = detection["id"]
            bbox = detection["bbox"]  # [u, v, w, h]
            
            u_center = bbox[0] + bbox[2] / 2
            v_center = bbox[1] + bbox[3] / 2
            
            # Use bbox-based depth estimation (better than placeholder depth map)
            depth = self.depth_estimator.estimate_depth_from_bbox(
                tuple(bbox), rgb_image.shape[0]
            )
            
            x, y, z = self.coord_transformer.pixel_to_world(
                u_center, v_center, depth, drone_pose
            )
            
            objects.append({
                "id": obj_id,
                "type": "target",
                "bbox": bbox,
                "estimated_coords": [round(x*10, 2), round(y*10, 2), round(max(z*10, 0), 2)]
            })
        
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
        """Simple rule-based goal inference from instruction."""
        if not objects:
            return "()" 
            
        obj_id = objects[0]["id"]
        return f"(scanned {obj_id})"
    
    def process_flight_plan(
        self,
        flight_plan_path: str,
        rgb_image: np.ndarray,
        drone_pose: DronePose
    ) -> Dict:
        """
        Process flight plan JSON: read objects, compute 3D coordinates, update file.
        """
        flight_plan_file = Path(flight_plan_path)
        
        if not flight_plan_file.exists():
            raise FileNotFoundError(f"Flight plan file not found: {flight_plan_path}")
        
        with open(flight_plan_file, 'r') as f:
            flight_plan = json.load(f)
        
        objects = flight_plan.get("objects", [])
        
        for obj in objects:
            if "bbox" in obj:
                bbox = obj["bbox"]  # [u, v, w, h]
                
                u_center = bbox[0] + bbox[2] / 2
                v_center = bbox[1] + bbox[3] / 2
                
                # Use bbox-based depth estimation
                depth = self.depth_estimator.estimate_depth_from_bbox(
                    tuple(bbox), rgb_image.shape[0]
                )
                
                x, y, z = self.coord_transformer.pixel_to_world(
                    u_center, v_center, depth, drone_pose
                )
                
                obj["estimated_coords"] = [round(x, 2), round(y, 2), round(max(z, 0), 2)]
                obj["estimated_depth"] = round(depth * 10, 2)
        
        with open(flight_plan_file, 'w') as f:
            json.dump(flight_plan, f, indent=4)
        
        print(f"✓ Updated flight plan with 3D coordinates: {flight_plan_path}")
        
        return flight_plan


if __name__ == "__main__":
    print("Perception Pipeline Library")
    print("Import this module to use PerceptionPipeline, DronePose, etc.")
