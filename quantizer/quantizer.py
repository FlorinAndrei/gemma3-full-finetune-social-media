from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
import torch

# from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.entrypoints import oneshot

MODEL_ID = "google/gemma-3-12b-it"
OUTPUT_DIR = "model_gemma3-12b-it-4bit"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = AWQModifier(
    ignore=["lm_head"],
    # Offload cached activations to CPU to reduce GPU memory usage
    offload_device=torch.device("cpu"),
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": False,
                "strategy": "group",
                # Changed from 128 to 16 to be divisible by 4304
                "group_size": 16,
            },
        }
    },
)

# Calibration dataset - use a representative sample
CALIBRATION_DATASET = "ultrachat-200k"
# bigger is better, but will require more RAM
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 2048

# Run quantization
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=CALIBRATION_DATASET,
    splits="train",
    recipe=recipe,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQ_LENGTH,
    preprocessing_num_workers=os.cpu_count(),
    output_dir=OUTPUT_DIR,
)

# Save processor for completeness
processor.save_pretrained(OUTPUT_DIR)

print(f"Quantized model saved to {OUTPUT_DIR}")
