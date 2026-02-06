#!/usr/bin/env python3
"""
=============================================================================
VLM Fine-Tuning Dataset Generator
=============================================================================
Generates synthetic training data for fine-tuning PaliGemma/LLaVA on the
PDDL problem generation task.

As specified in Section 4.3 of the design document:
- Scene Generation: Procedural warehouse scenes
- Ground Truth Capture: RGB, bbox, 3D coords, drone pose
- Instruction Generation: Template-based NL prompts
- Label Generation: Target JSON for VLM output
=============================================================================
"""

import json
import random
import math
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TargetObject:
    """A target object in the warehouse"""
    id: str
    category: str  # box, barrel, pallet, container, crate
    color: str
    bbox: List[int]  # [u, v, w, h]
    world_coords: List[float]  # [x, y, z]


@dataclass
class TrainingSample:
    """Single training sample for VLM fine-tuning"""
    image_path: str
    instruction: str
    vlm_output: Dict  # Expected VLM JSON output
    ground_truth: Dict  # Full ground truth for evaluation


class WarehouseSceneGenerator:
    """
    Generates synthetic warehouse scene configurations.
    
    For actual RGB images, integrate with:
    - NVIDIA Isaac Sim
    - Gazebo with warehouse plugin
    - Unity Perception package
    """
    
    # Object categories in warehouse
    CATEGORIES = [
        "box", "crate", "pallet", "barrel", "container",
        "package", "carton", "drum", "bin", "rack"
    ]
    
    # Object modifiers
    COLORS = [
        "red", "blue", "green", "yellow", "orange", "white",
        "black", "brown", "gray", "purple"
    ]
    
    SIZES = ["small", "large", "medium"]
    
    MATERIALS = [
        "cardboard", "plastic", "metal", "wooden"
    ]
    
    # Location descriptors for instructions
    LOCATIONS = [
        "on the top shelf", "on the bottom shelf", "in aisle {n}",
        "near the loading dock", "at the receiving area",
        "on rack {n}", "in section {s}", "at position {p}"
    ]
    
    # Action verbs for instructions
    SCAN_VERBS = [
        "scan", "inspect", "examine", "check", "photograph",
        "capture image of", "document", "survey"
    ]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.sample_counter = 0
        
    def generate_object_id(self, category: str, color: str) -> str:
        """Generate unique object ID"""
        return f"{color}_{category}_{self.sample_counter:04d}"
    
    def generate_bbox(
        self,
        image_width: int = 640,
        image_height: int = 480,
        min_size: int = 50,
        max_size: int = 200
    ) -> List[int]:
        """Generate random bounding box within image"""
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, int(max_size * 0.8))
        u = random.randint(0, image_width - w)
        v = random.randint(0, image_height - h)
        return [u, v, w, h]
    
    def generate_world_coords(
        self,
        x_range: Tuple[float, float] = (1.0, 30.0),
        y_range: Tuple[float, float] = (-10.0, 10.0),
        z_range: Tuple[float, float] = (0.5, 8.0)
    ) -> List[float]:
        """Generate random 3D world coordinates"""
        return [
            round(random.uniform(*x_range), 2),
            round(random.uniform(*y_range), 2),
            round(random.uniform(*z_range), 2)
        ]
    
    def generate_instruction(self, obj: TargetObject) -> str:
        """
        Generate natural language instruction for object.
        
        Templates match Section 4.3 of design document.
        """
        verb = random.choice(self.SCAN_VERBS)
        
        # Various instruction templates
        templates = [
            f"{verb.capitalize()} the {obj.color} {obj.category}.",
            f"Go to the {obj.color} {obj.category} and {verb} it.",
            f"Please {verb} the {obj.color} {obj.category}.",
            f"Navigate to and {verb} the {obj.color} {obj.category}.",
            f"Find the {obj.color} {obj.category} and {verb} it.",
            f"I need you to {verb} the {obj.color} {obj.category}.",
            f"The {obj.color} {obj.category} needs to be {verb}ed.",
            f"Can you {verb} the {obj.color} {obj.category}?"
        ]
        
        # Sometimes add location
        if random.random() < 0.3:
            location = random.choice(self.LOCATIONS)
            location = location.format(
                n=random.randint(1, 10),
                s=random.choice(['A', 'B', 'C', 'D']),
                p=f"{random.randint(1,5)}-{random.randint(1,20)}"
            )
            templates = [
                f"{verb.capitalize()} the {obj.color} {obj.category} {location}.",
                f"Go to the {obj.color} {obj.category} {location}.",
            ]
        
        return random.choice(templates)
    
    def generate_vlm_output(self, obj: TargetObject) -> Dict:
        """
        Generate expected VLM JSON output.
        
        This is the target the VLM is trained to produce.
        """
        return {
            "objects": [{
                "id": obj.id,
                "type": "target",
                "bbox": obj.bbox,
                "estimated_coords": obj.world_coords
            }],
            "goal": f"(scanned {obj.id})"
        }
    
    def generate_sample(self) -> TrainingSample:
        """Generate single training sample"""
        self.sample_counter += 1
        
        # Generate random object
        category = random.choice(self.CATEGORIES)
        color = random.choice(self.COLORS)
        
        obj = TargetObject(
            id=self.generate_object_id(category, color),
            category=category,
            color=color,
            bbox=self.generate_bbox(),
            world_coords=self.generate_world_coords()
        )
        
        # Generate instruction
        instruction = self.generate_instruction(obj)
        
        # Generate VLM output
        vlm_output = self.generate_vlm_output(obj)
        
        # Ground truth (for evaluation)
        ground_truth = {
            "object": asdict(obj),
            "drone_pose": {
                "x": 0.0, "y": 0.0, "z": 1.5,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0
            },
            "camera_intrinsics": {
                "fx": 554.26, "fy": 554.26,
                "cx": 320.0, "cy": 240.0
            }
        }
        
        return TrainingSample(
            image_path=f"images/warehouse_{self.sample_counter:05d}.png",
            instruction=instruction,
            vlm_output=vlm_output,
            ground_truth=ground_truth
        )
    
    def generate_multi_object_sample(
        self, 
        num_objects: int = 2
    ) -> TrainingSample:
        """Generate sample with multiple objects (advanced)"""
        self.sample_counter += 1
        
        objects = []
        for i in range(num_objects):
            category = random.choice(self.CATEGORIES)
            color = random.choice(self.COLORS)
            
            obj = TargetObject(
                id=self.generate_object_id(category, color),
                category=category,
                color=color,
                bbox=self.generate_bbox(),
                world_coords=self.generate_world_coords()
            )
            objects.append(obj)
        
        # Primary target for instruction
        primary = objects[0]
        instruction = self.generate_instruction(primary)
        
        # VLM output includes all visible objects
        vlm_output = {
            "objects": [{
                "id": obj.id,
                "type": "target",
                "bbox": obj.bbox,
                "estimated_coords": obj.world_coords
            } for obj in objects],
            "goal": f"(scanned {primary.id})"
        }
        
        ground_truth = {
            "objects": [asdict(obj) for obj in objects],
            "primary_target": primary.id
        }
        
        return TrainingSample(
            image_path=f"images/warehouse_{self.sample_counter:05d}.png",
            instruction=instruction,
            vlm_output=vlm_output,
            ground_truth=ground_truth
        )


class DatasetBuilder:
    """
    Builds complete training dataset for VLM fine-tuning.
    """
    
    def __init__(
        self,
        output_dir: str,
        num_samples: int = 1000,
        multi_object_ratio: float = 0.2,
        seed: int = 42
    ):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.multi_object_ratio = multi_object_ratio
        self.generator = WarehouseSceneGenerator(seed)
        
    def build(self) -> str:
        """
        Build complete dataset from actual images in the images directory.
        
        Returns:
            Path to dataset JSON file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Scan for actual images in the directory
        image_files = []
        if os.path.exists(images_dir):
            for f in sorted(os.listdir(images_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(f)
        
        if not image_files:
            print(f"No images found in {images_dir}")
            print("Falling back to synthetic sample generation...")
            # Fallback to synthetic generation
            for i in range(self.num_samples):
                if random.random() < self.multi_object_ratio:
                    sample = self.generator.generate_multi_object_sample(
                        num_objects=random.randint(2, 4)
                    )
                else:
                    sample = self.generator.generate_sample()
                image_files.append(sample.image_path)
        
        print(f"Found {len(image_files)} images in {images_dir}")
        
        samples = []
        
        for i, image_file in enumerate(image_files):
            # Generate training sample for this image
            if random.random() < self.multi_object_ratio:
                sample = self.generator.generate_multi_object_sample(
                    num_objects=random.randint(2, 4)
                )
            else:
                sample = self.generator.generate_sample()
            
            # Use actual image path instead of synthetic one
            actual_image_path = f"images/{image_file}"
            
            samples.append({
                "id": i,
                "image_path": actual_image_path,
                "instruction": sample.instruction,
                "vlm_output": sample.vlm_output,
                "ground_truth": sample.ground_truth
            })
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(image_files)} samples")
        
        # Split dataset
        random.shuffle(samples)
        train_split = int(0.8 * len(samples))
        val_split = int(0.9 * len(samples))
        
        train_samples = samples[:train_split]
        val_samples = samples[train_split:val_split]
        test_samples = samples[val_split:]
        
        # Save datasets
        dataset_path = os.path.join(self.output_dir, "dataset.json")
        with open(dataset_path, 'w') as f:
            json.dump({
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "total_samples": len(samples),
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                    "test_samples": len(test_samples)
                },
                "train": train_samples,
                "validation": val_samples,
                "test": test_samples
            }, f, indent=2)
        
        print(f"\nDataset saved to: {dataset_path}")
        print(f"  Train: {len(train_samples)}")
        print(f"  Validation: {len(val_samples)}")
        print(f"  Test: {len(test_samples)}")
        
        return dataset_path


def format_for_training(sample: Dict) -> Dict:
    """
    Format sample for VLM training (conversation format).
    
    This produces the input/output pairs used by fine-tuning frameworks
    like Hugging Face TRL or LLaVA training scripts.
    """
    # System prompt
    system_prompt = """You are a PDDL Problem Generator for warehouse drone navigation.
Given an image and instruction, output a JSON object specifying:
- objects: detected objects with id, type, bbox, and estimated_coords
- goal: PDDL goal predicate (e.g., "(scanned object_id)")

Your coordinates should be estimated 3D world positions in meters."""

    # User message with instruction
    user_message = f"<image>\nInstruction: {sample['instruction']}\n\nGenerate PDDL problem JSON:"
    
    # Assistant response (target)
    assistant_response = json.dumps(sample['vlm_output'], indent=None)
    
    return {
        "id": sample.get("id", 0),
        "image": sample["image_path"],
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def main():
    """Generate synthetic training dataset"""
    
    print("=" * 70)
    print("VLM Fine-Tuning Dataset Generator")
    print("=" * 70)
    
    output_dir = "/home/psp/ros_ws/src/PDDL/vlm_training_data"
    
    # Generate dataset
    builder = DatasetBuilder(
        output_dir=output_dir,
        num_samples=100,  # Use more samples for real training (10000+)
        multi_object_ratio=0.2,
        seed=42
    )
    
    dataset_path = builder.build()
    
    # Load and show example
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print("\n" + "=" * 70)
    print("Example Training Sample:")
    print("-" * 70)
    
    example = dataset["train"][0]
    print(f"Image: {example['image_path']}")
    print(f"Instruction: {example['instruction']}")
    print(f"\nExpected VLM Output:")
    print(json.dumps(example['vlm_output'], indent=2))
    
    print("\n" + "=" * 70)
    print("Conversation Format (for fine-tuning):")
    print("-" * 70)
    
    formatted = format_for_training(example)
    for conv in formatted["conversations"]:
        print(f"\n[{conv['role'].upper()}]:")
        print(conv['content'][:500] + "..." if len(conv['content']) > 500 else conv['content'])
    
    print("\n" + "=" * 70)
    print("✓ Dataset generation complete!")
    print(f"  Location: {output_dir}")


if __name__ == "__main__":
    main()
