# Warehouse Drone Navigation: Neuro-Symbolic PDDL + VLM System

Complete implementation of autonomous drone navigation in warehouse environments using **PDDL 2.1 planning** with **Vision Language Model** perception.

> **Key Innovation**: No predefined waypoints — locations are dynamically generated from visual perception at runtime.

## 📁 File Overview 

| File | Description |
|------|-------------|
| **Core PDDL** ||
| `warehouse_drone_domain.pddl` | PDDL 2.1 domain with durative actions, numeric fluents |
| `example_problem_scan.pddl` | Example problem demonstrating VLM-generated coordinates |
| **VLM Integration** ||
| `vlm_pddl_generator.py` | Converts VLM JSON output → valid PDDL problem files |
| `perception_pipeline.py` | Depth estimation + coordinate transformation (2D→3D) |
| `vlm_dataset_generator.py` | Generates synthetic training data for VLM fine-tuning |
| `vlm_finetuning.py` | LoRA fine-tuning script for PaliGemma/LLaVA |
| **ROS 2 Integration** ||
| `ros2_vlm_pddl_node.py` | ROS 2 node bridging VLM perception with PlanSys2 |
| **Documentation** ||
| `design_document.txt` | Extracted text from original design document |
| `README.md` | This file |

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Drone Camera   │────▶│  Fine-tuned VLM  │────▶│  PDDL       │
│  (RGB Image)    │     │  (PaliGemma)     │     │  Problem    │
└─────────────────┘     └──────────────────┘     └──────┬──────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  User Command   │────▶│  Depth Network   │     │  PDDL       │
│  (Natural Lang) │     │  (MiDaS/DA)      │     │  Planner    │
└─────────────────┘     └────────┬─────────┘     │  (POPF)     │
                                 │               └──────┬──────┘
                                 ▼                      │
                        ┌──────────────────┐           │
                        │  3D Coordinates  │           │
                        │  (x, y, z)       │           ▼
                        └──────────────────┘     ┌─────────────┐
                                                 │  ROS 2      │
                                                 │  Executor   │
                                                 └──────┬──────┘
                                                        │
                                                        ▼
                                                 ┌─────────────┐
                                                 │   Drone     │
                                                 │   Actions   │
                                                 └─────────────┘
```

---

## 🚀 Quick Start 

### 1. Test PDDL Domain with Planner

```bash
# Install POPF planner
sudo apt install popf

# Run planner
cd /home/psp/ros_ws/src/PDDL
popf warehouse_drone_domain.pddl example_problem_scan.pddl
```

### 2. Generate VLM Training Data

```bash
python3 vlm_dataset_generator.py
# Creates vlm_training_data/dataset.json with 100 samples
```

### 3. Test VLM → PDDL Pipeline

```bash
python3 vlm_pddl_generator.py
# Generates problem from simulated VLM output
```

### 4. Test Perception Pipeline

```bash
python3 perception_pipeline.py
# Demonstrates depth estimation + coordinate transformation
```

---

## 📋 PDDL Domain Features

### Durative Actions (Time-Dependent)

| Action | Duration | Description |
|--------|----------|-------------|
| `take_off` | 5s | Ground → hover state |
| `fly_to_target` | distance/speed | Navigate to VLM-detected object |
| `scan_target` | 4s | Capture imagery (proximity required) |
| `land` | 5s | Controlled descent |
| `recharge` | variable | Battery refill when docked |

### Numeric Fluents (Continuous State)

```pddl
;; 3D Drone Position
(x ?d - drone)  (y ?d - drone)  (z ?d - drone)

;; Target Coordinates (VLM output)
(tx ?t - target)  (ty ?t - target)  (tz ?t - target)

;; Battery Management
(battery-level ?d)  (discharge-rate-fly ?d)  (discharge-rate-hover ?d)

;; Distance (precomputed by perception)
(distance-to ?d - drone ?t - target)
```

---

## 🧠 VLM Fine-Tuning Pipeline

### Training Data Format

```json
{
  "image_path": "images/warehouse_00001.png",
  "instruction": "Scan the red box on the top shelf",
  "vlm_output": {
    "objects": [{
      "id": "red_box",
      "type": "target",
      "bbox": [320, 240, 100, 80],
      "estimated_coords": [12.5, 3.2, 4.0]
    }],
    "goal": "(scanned red_box)"
  }
}
```

### Fine-Tuning Configuration (LoRA)

```python
lora_config = LoraConfig(
    r=16,                              # Rank
    lora_alpha=32,                     # Alpha scaling
    target_modules=["q_proj", "v_proj"], # Attention layers
    lora_dropout=0.05
)
```

### Grammar-Constrained Decoding

The fine-tuned VLM uses **vLLM** or **Outlines** for structured output:
- Model is mathematically constrained to produce valid JSON
- Schema enforcement during generation (not post-processing)

---

## 🔄 ROS 2 Integration

### Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/vlm_pddl/instruction` | `String` | Input: NL command |
| `/vlm_pddl/generated_problem` | `String` | Output: PDDL problem |
| `/vlm_pddl/status` | `String` | Output: Status updates |

### Run the Node

```bash
# Terminal 1: Launch drone simulation
ros2 launch sjtu_drone_bringup sjtu_drone_bringup.launch.py

# Terminal 2: Run VLM PDDL Node
python3 ros2_vlm_pddl_node.py

# Terminal 3: Send instruction
ros2 topic pub /vlm_pddl/instruction std_msgs/String "data: 'Scan the red box'"
```

### PlanSys2 Integration

The node automatically updates PlanSys2 Knowledge Base:
```python
# Add detected object
add_instance("red_box", "target")

# Set coordinates
update_function("tx", "red_box", 12.5)
update_function("ty", "red_box", 3.2)
update_function("tz", "red_box", 4.0)
```

---

## 📊 Perception Pipeline

### Depth Estimation

Uses MiDaS or Depth Anything to convert 2D images to depth maps:

```python
depth_estimator = DepthEstimator("midas_v21_small")
depth_map = depth_estimator.estimate_depth(rgb_image)
d_obj = depth_estimator.sample_depth_in_bbox(depth_map, bbox)
```

### Coordinate Transformation (Pinhole Model)

```python
# 2D pixel + depth → 3D camera frame
X_cam = (u - cx) * depth / fx
Y_cam = (v - cy) * depth / fy
Z_cam = depth

# Camera frame → World frame
P_world = R_drone @ P_cam + T_drone
```

---

## 📝 Expected PDDL Plan Output

For instruction: "Scan the red box"

```
0.000: (undock drone1)  [2.000]
2.000: (take_off drone1)  [5.000]
7.000: (fly_to_target drone1 red_box)  [6.750]
13.750: (scan_target drone1 red_box)  [4.000]
17.750: (fly_to_zone drone1 charging_station)  [6.750]
24.500: (land drone1)  [5.000]
```

---

## 🔧 Dependencies

### Python
```bash
pip install numpy opencv-python transformers peft datasets accelerate
pip install torch  # With CUDA support if available

# For grammar-constrained decoding
pip install vllm  # or: pip install outlines
```

### ROS 2
```bash
sudo apt install ros-humble-plansys2*
sudo apt install popf  # Or install OPTIC/ENHSP
```

---

## 📚 References

- [PDDL 2.1 Specification](https://planning.wiki/ref/pddl21/domain)
- [PlanSys2 Documentation](https://plansys2.github.io/)
- [PaliGemma Fine-Tuning Guide](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_object_detection_grounding)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything)

---

## ⚡ Next Steps

1. **Collect Real Images**: Gather warehouse footage with ground truth
2. **Fine-Tune VLM**: Train on 10k+ samples with domain-specific data
3. **Integrate Depth Model**: Replace placeholder with MiDaS/Depth Anything
4. **Test with PlanSys2**: Full planning and execution loop
5. **Deploy on Drone**: Real-world testing in warehouse environment
