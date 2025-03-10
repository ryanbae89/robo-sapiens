# inference/modules/model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(base_model_id, lora_path=None, quantization_config=None):
    """
    Loads a base model + optional LoRA adapter.
    Allows for 4-bit quantization if specified in quantization_config.
    Returns (model, tokenizer).
    """
    # Check for GPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = {"": 0}
    # print(f"DEVICE: {device_map}")
    
    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # 2) Check if we need to load in 4-bit
    if quantization_config and quantization_config.get("load_in_4bit", False):
        
        # Build bitsandbytes config
        bnb_4bit_quant_type = quantization_config.get("bnb_4bit_quant_type", "nf4")
        bnb_4bit_compute_dtype = quantization_config.get("bnb_4bit_compute_dtype", "float16")
        bnb_4bit_use_double_quant = quantization_config.get("bnb_4bit_use_double_quant", False)
        
        # Convert string to actual dtype
        if bnb_4bit_compute_dtype == "float16":
            compute_dtype = torch.float16
        elif bnb_4bit_compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = None  # fallback, typically float32
        
        # Set up bnb config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        # 3) Otherwise load standard FP16 or full precision
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map=device_map,
            trust_remote_code=True,
        )
    
    # 4) If there's a LoRA adapter, load it
    if lora_path:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map=device_map,
        )
    
    # 5) Put model in eval mode
    model.eval()
    
    return model, tokenizer
