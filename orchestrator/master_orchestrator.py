# orchestrator/master_orchestrator.py

import os
import yaml
import uuid

from data_processing.scripts.whatsapp_process_pipeline import run_whatsapp_pipeline
from training.scripts.train import run_training_pipeline
from inference.scripts.generate_test_conversation import run_inference_test

def run_full_pipeline(
    orchestrator_config,
):
    """
    High-level orchestrator that runs:
      1) Data Processing
      2) Training
      3) Inference
    in sequence using a single config file.
    """
    
    # Parse orchestrator config file
    print(f"[run_full_pipeline] Loading config file: {orchestrator_config}")
    with open(orchestrator_config, "r") as f:
        config = yaml.safe_load(f)
        
    # Generate orid (orchestrator run id)
    or_id = str(uuid.uuid4())
    print(f"ORCHESTRATOR RUN ID: {or_id}")
    
    # ----------------------------
    # 1. Data Processing
    # ----------------------------
    data_processing_pipeline_params = config["data_processing_pipeline_params"]
    data_processing_pipeline_params["output_dir"] = os.path.join(data_processing_pipeline_params["output_dir"], or_id)
    
    # Open data processing config
    with open(data_processing_pipeline_params["config"], "r") as f:
        data_processing_config = yaml.safe_load(f)
    
    # Call data processing pipeline
    run_whatsapp_pipeline(
        config = data_processing_pipeline_params["config"],
        raw_input_file = data_processing_pipeline_params["raw_input_file"],
        output_dir = data_processing_pipeline_params["output_dir"],
        chat_info = config["chat_info"]
    )
    
    # ----------------------------
    # 2. Training
    # ----------------------------
    training_pipeline_params = config["training_pipeline_params"]
    training_pipeline_params["output_dir"] = os.path.join(training_pipeline_params["output_dir"], or_id)
    processed_data = os.path.join(data_processing_pipeline_params["output_dir"], "messages_processed.json")
    
    # Load training config file
    with open(training_pipeline_params["config"], "r") as f:
        training_config = yaml.safe_load(f)
    
    # Call training pipeline
    run_training_pipeline(
        config = training_pipeline_params["config"],
        processed_data = processed_data,
        output_dir = training_pipeline_params["output_dir"]
    )

    # ----------------------------
    # 3. Inference
    # ----------------------------
    inference_test_params = config["inference_test_params"]
    test_context = inference_test_params["test_speaker"] + ": " + inference_test_params["test_prompt"]

    # Example: your pipeline might specify two config paths for model + generation
    model_config_path = inference_test_params["model_config"]
    generation_config_path = inference_test_params["generation_config"]
    
    # Update the model config file to point to the newly trained LoRA output_dir and quantization params
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f) 
    model_config["model"]["lora_path"] = training_pipeline_params["output_dir"]
    model_config["quantization"] = training_config["quantization"]
    with open(model_config_path, "w") as f:
        yaml.safe_dump(model_config, f)
        
    # Update the generation config file with prompt information from orchestrator config
    with open(generation_config_path, "r") as f:
        generation_config = yaml.safe_load(f)
    generation_config["prompt"]["group_description"] = config["chat_info"]["group_description"]
    generation_config["prompt"]["group_members"] = config["chat_info"]["group_members"]
    with open(generation_config_path, "w") as f:
        yaml.safe_dump(generation_config, f)
        
    # Call the inference function
    print("[run_full_pipeline] Running inference test...")
    full_convo = run_inference_test(
        model_config_path=model_config_path,
        generation_config_path=generation_config_path,
        context=test_context
    )

    print("\n=== FULL CONVERSATION OUTPUT ===")
    print(full_convo)
    print("=== Pipeline complete! ===")
    
    # --------------------------------
    # 4. Dump All Configs to One File
    # --------------------------------
    # Suppose we want to store them in the same directory as training outputs
    final_config_output_dir = training_pipeline_params["output_dir"]
    final_run_config_path = os.path.join(final_config_output_dir, "orchestrator_run_config.yaml")

    # Build a single dictionary that consolidates everything
    final_run_config = {
        "orchestrator_run_id": or_id,
        "orchestrator_config": config,
        "data_processing_config": data_processing_config,
        "training_config": training_config,
        "model_config": model_config,
        "generation_config": generation_config,
    }

    with open(final_run_config_path, "w") as f:
        yaml.safe_dump(final_run_config, f, sort_keys=False)

    print(f"[run_full_pipeline] Final run config saved to: {final_run_config_path}")
    

def main():
    # Optionally parse args here or define them inline
    orchestrator_config = "/mnt/c/Users/ryanb/Work/Projects/robo-sapiens/orchestrator/configs/full_orchestrator_config.yaml"

    run_full_pipeline(
        orchestrator_config
    )

if __name__ == "__main__":
    main()
