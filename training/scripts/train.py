# training/scripts/train_llama.py

import os
import wandb
import yaml

from training.modules.data_utils import load_and_split_data
from training.modules.prompt_utils import format_prompt
from training.modules.training_pipeline import run_training
from peft import PeftModel

def run_training_pipeline(config: str,
                          processed_data: str,
                          output_dir: str):
    """
    Train a LLaMA model with LoRA using a given config dict.
    Returns the trainer or the output directory containing LoRA adapter weights.
    """
    # --- Load config from YAML ---
    print(f"[run_training_pipeline] Loading config file: {config}")
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    
    # 1. Initialize W&B (optional)
    wandb.login()
    wandb.init(project="robo-sapiens")

    # 2. Extract config fields
    model_id = config["model"].get("model_id", "meta-llama/Llama-2-7b-chat-hf")

    # Data info
    sample_size = config["data"].get("sample_size", 100)
    test_ratio = config["data"].get("test_ratio", 0.1)

    # Training params
    epochs = config["training"].get("epochs", 2)
    learning_rate = float(config["training"].get("learning_rate", 2e-4))
    batch_size = config["training"].get("batch_size", 4)
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 4)
    
    # 3. Load data
    data = load_and_split_data(
        json_file_path=processed_data,
        sample_size=sample_size,
        test_ratio=test_ratio
    )
    print(f"[run_training_pipeline] Dataset shape: {data.shape}")

    # 4. Run training pipeline
    trainer = run_training(
        model_id=model_id,
        data=data,
        format_func=format_prompt,
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        quantization_params=config["quantization"]
    )

    # 5. Save ONLY the LoRA adapter weights
    if isinstance(trainer.model, PeftModel):
        print("[run_training_pipeline] Saving LoRA adapter weights only...")
        trainer.model.save_pretrained(output_dir)
        print(f"[run_training_pipeline] LoRA adapter weights saved to: {output_dir}")
    else:
        print(f"[run_training_pipeline] LoRA adapter weights saved at {output_dir}")

    return trainer

# (Optional) If you still want to allow CLI usage, you can define:
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a LLaMA model with LoRA via config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config file.")
    parser.add_argument("processed_data", type=str, required=True, help="Path to the processed output of data_processing pipeline step." )
    args = parser.parse_args()

    # Load YAML from file
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Call our new function
    run_training_pipeline(config_dict)

if __name__ == "__main__":
    main()