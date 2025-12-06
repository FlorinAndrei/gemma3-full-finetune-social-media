from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
import torch

# from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.entrypoints import oneshot

MODEL_ID = "google/gemma-3-12b-it"
OUTPUT_DIR = "gemma3-12b-it-4bit"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

"""
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",  # 4-bit weights, 16-bit activations
    ignore=["lm_head"],  # Keep output layer in full precision
)
"""

recipe = AWQModifier(
    ignore=["lm_head"],
    offload_device=torch.device("cpu"),  # Offload cached activations to CPU to reduce GPU memory usage
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": False,
                "strategy": "group",
                "group_size": 16,  # Changed from 128 to 16 to be divisible by 4304
            }
        }
    }
)

# Calibration dataset - use a representative sample
CALIBRATION_DATASET = "ultrachat-200k"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 2048

# Run quantization
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=CALIBRATION_DATASET,
    splits="train",  # Use the train split for calibration
    recipe=recipe,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQ_LENGTH,
    preprocessing_num_workers=os.cpu_count(),
    output_dir=OUTPUT_DIR,
)

# Save processor for completeness
processor.save_pretrained(OUTPUT_DIR)

print(f"Quantized model saved to {OUTPUT_DIR}")
