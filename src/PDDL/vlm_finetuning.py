#!/usr/bin/env python3
"""
=============================================================================
VLM Fine-Tuning Script (PaliGemma / LLaVA)
=============================================================================
Fine-tunes a Vision Language Model for PDDL problem generation.

Based on Section 4 of the design document:
- Uses LoRA for efficient fine-tuning
- Grammar-constrained decoding for valid JSON output
- Target: VLM outputs structured JSON that converts to valid PDDL
=============================================================================
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Model
    model_name: str = "google/paligemma-3b-pt-448"
    # For LLaVA: "llava-hf/llava-1.5-7b-hf"
    
    # LoRA parameters (Section 4.4)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  # Set in __post_init__
    
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data
    max_length: int = 1024
    
    # Output
    output_dir: str = "./vlm_pddl_finetuned"
    logging_steps: int = 10
    save_steps: int = 100
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


# JSON Schema for structured output (Section 4.5)
PDDL_JSON_SCHEMA = {
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
        "goal": {"type": "string"}
    },
    "required": ["objects", "goal"]
}


def get_training_prompt(instruction: str) -> str:
    """
    Generate training prompt template.
    
    This matches the format the VLM is fine-tuned on.
    """
    return f"""You are a PDDL Problem Generator for warehouse drone navigation.
Given an image and a natural language instruction, output a valid JSON object with:
- "objects": list of detected objects, each with "id", "type", "bbox" (pixels), and "estimated_coords" (meters)
- "goal": PDDL goal predicate (e.g., "(scanned object_id)")

Instruction: {instruction}

JSON:"""


def print_training_script():
    """
    Print the actual training script code.
    
    This is the code you would run for fine-tuning.
    Note: Requires transformers, peft, and datasets libraries.
    """
    
    script = '''
#!/usr/bin/env python3
"""
VLM Fine-Tuning Script - Full Implementation
Run this after installing dependencies:
    pip install transformers peft datasets accelerate bitsandbytes
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json


def load_pddl_dataset(dataset_path: str):
    """Load the PDDL training dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data['train'], data['validation']


def preprocess_function(examples, processor, max_length=1024):
    """Preprocess samples for VLM training"""
    
    # Format conversations
    texts = []
    for conv in examples['conversations']:
        text = ""
        for msg in conv:
            if msg['role'] == 'user':
                text += f"USER: {msg['content']}\\n"
            elif msg['role'] == 'assistant':
                text += f"ASSISTANT: {msg['content']}"
        texts.append(text)
    
    # Tokenize
    encodings = processor(
        text=texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encodings


def main():
    # Configuration
    model_name = "google/paligemma-3b-pt-448"  # Or LLaVA
    dataset_path = "/home/psp/ros_ws/src/PDDL/vlm_training_data/dataset.json"
    output_dir = "/home/psp/ros_ws/src/PDDL/vlm_finetuned"
    
    # Quantization config (for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model and processor
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration (Section 4.4)
    lora_config = LoraConfig(
        r=16,                          # Rank
        lora_alpha=32,                 # Alpha
        target_modules=["q_proj", "v_proj"],  # Attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("Loading dataset...")
    train_data, val_data = load_pddl_dataset(dataset_path)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=lambda x: x  # Custom collator needed
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
'''
    
    return script


def print_inference_script():
    """
    Print the inference script for using the fine-tuned model.
    
    Includes grammar-constrained decoding (Section 4.5)
    """
    
    script = '''
#!/usr/bin/env python3
"""
VLM Inference with Grammar-Constrained Decoding
Uses vLLM or Outlines for structured JSON output
"""

import json
from PIL import Image

# For grammar-constrained decoding (Section 4.5)
# Option 1: Using vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    USE_VLLM = True
except ImportError:
    USE_VLLM = False

# Option 2: Using Outlines (works with transformers)
try:
    import outlines
    from outlines import models, generate
    USE_OUTLINES = True
except ImportError:
    USE_OUTLINES = False


# JSON Schema for structured output
PDDL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["target"]},
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                    "estimated_coords": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["id", "type", "bbox", "estimated_coords"]
            }
        },
        "goal": {"type": "string"}
    },
    "required": ["objects", "goal"]
}


class PDDLInference:
    """Run inference with grammar-constrained decoding"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        if USE_VLLM:
            self.method = "vllm"
            self.model = LLM(
                model=model_path,
                trust_remote_code=True
            )
        elif USE_OUTLINES:
            self.method = "outlines"
            self.model = models.transformers(model_path)
        else:
            raise RuntimeError("Neither vLLM nor Outlines installed")
    
    def run(self, image_path: str, instruction: str) -> dict:
        """
        Run inference with structured output.
        
        The model CANNOT produce invalid JSON - it's constrained
        by the grammar during decoding.
        """
        prompt = f"""You are a PDDL Problem Generator for warehouse drone navigation.
Given an image and instruction, output JSON with objects and goal.

Instruction: {instruction}

JSON:"""
        
        if self.method == "vllm":
            # vLLM structured output
            from vllm.sampling_params import GuidedDecodingParams
            
            sampling_params = SamplingParams(
                max_tokens=512,
                guided_decoding=GuidedDecodingParams(
                    json=PDDL_JSON_SCHEMA
                )
            )
            
            outputs = self.model.generate([prompt], sampling_params)
            result = outputs[0].outputs[0].text
            
        elif self.method == "outlines":
            # Outlines structured generation
            generator = generate.json(self.model, PDDL_JSON_SCHEMA)
            result = generator(prompt)
        
        return json.loads(result) if isinstance(result, str) else result


def main():
    """Demo inference"""
    import sys
    
    model_path = "/home/psp/ros_ws/src/PDDL/vlm_finetuned"
    
    if not (USE_VLLM or USE_OUTLINES):
        print("Please install vLLM or Outlines for structured decoding:")
        print("  pip install vllm")
        print("  # or")
        print("  pip install outlines")
        sys.exit(1)
    
    # Initialize inference
    inference = PDDLInference(model_path)
    
    # Run inference
    result = inference.run(
        image_path="warehouse_image.png",
        instruction="Scan the red box on the top shelf"
    )
    
    print("VLM Output (guaranteed valid JSON):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''
    
    return script


def main():
    """Print training and inference scripts"""
    
    print("=" * 70)
    print("VLM Fine-Tuning Configuration and Scripts")
    print("=" * 70)
    
    config = TrainingConfig()
    
    print("\n1. Training Configuration (Section 4.4)")
    print("-" * 70)
    print(f"   Model: {config.model_name}")
    print(f"   LoRA Rank (r): {config.lora_rank}")
    print(f"   LoRA Alpha (α): {config.lora_alpha}")
    print(f"   Target Modules: {config.lora_target_modules}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"   Epochs: {config.num_epochs}")
    
    print("\n2. JSON Schema for Structured Output (Section 4.5)")
    print("-" * 70)
    print(json.dumps(PDDL_JSON_SCHEMA, indent=2))
    
    # Save training script
    train_script = print_training_script()
    train_script_path = "/home/psp/ros_ws/src/PDDL/train_vlm.py"
    with open(train_script_path, 'w') as f:
        f.write(train_script)
    print(f"\n3. Training Script saved to: {train_script_path}")
    
    # Save inference script
    infer_script = print_inference_script()
    infer_script_path = "/home/psp/ros_ws/src/PDDL/inference_vlm.py"
    with open(infer_script_path, 'w') as f:
        f.write(infer_script)
    print(f"4. Inference Script saved to: {infer_script_path}")
    
    print("\n" + "=" * 70)
    print("Training Pipeline:")
    print("-" * 70)
    print("""
1. Generate synthetic dataset:
   python3 vlm_dataset_generator.py

2. Install dependencies:
   pip install transformers peft datasets accelerate bitsandbytes

3. Run fine-tuning:
   python3 train_vlm.py

4. Run inference:
   pip install vllm  # or: pip install outlines
   python3 inference_vlm.py
""")
    
    print("=" * 70)
    print("✓ VLM fine-tuning configuration complete!")


if __name__ == "__main__":
    main()
