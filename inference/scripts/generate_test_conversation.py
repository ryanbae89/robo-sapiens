# inference/scripts/generate_test_conversation.py

import yaml
from inference.modules.model_loader import load_model
from inference.modules.inference_pipeline import generate_conversation

def run_inference_test(
    model_config_path: str,
    generation_config_path: str,
    context: str
) -> str:
    """
    Load configs, initialize model/tokenizer, generate a conversation
    from the given 'context'. Returns the final conversation string.
    """

    # 1) Load the model config
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # 2) Load the generation/inference config
    with open(generation_config_path, "r") as f:
        gen_config = yaml.safe_load(f)

    # 3) Extract relevant info
    base_model_id = model_config["model"]["base_model_id"]
    lora_path = model_config["model"].get("lora_path", None)
    quantization_config = model_config.get("quantization", {})

    # 4) Load model + tokenizer
    model, tokenizer = load_model(
        base_model_id=base_model_id,
        lora_path=lora_path,
        quantization_config=quantization_config
    )
    print("[run_inference_test] Model objects loaded successfully!")

    # 5) Generate text
    inference_params = gen_config["inference"]
    generation_params = gen_config["generation"]
    prompt_params = gen_config["prompt"]

    # 6) Generate the full conversation
    full_convo = generate_conversation(
        model=model,
        tokenizer=tokenizer,
        context=context,
        inference_params=inference_params,
        generation_params=generation_params,
        prompt_params=prompt_params
    )

    return full_convo

# Optional: keep a small CLI entry point if you still want to run it from command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test inference with two config files.")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to model config YAML (base model, LoRA path, quantization).")
    parser.add_argument("--generation-config", type=str, required=True,
                        help="Path to generation config YAML (temperature, max_new_tokens, etc.).")
    parser.add_argument("--context", type=str, default="Ryan Bae: guess what steve said...",
                        help="Context for conversation generation.")
    args = parser.parse_args()

    # Call our functional approach
    convo = run_inference_test(
        model_config_path=args.model_config,
        generation_config_path=args.generation_config,
        context=args.context
    )
    print("\n=== Full Conversation ===\n")
    print(convo)
