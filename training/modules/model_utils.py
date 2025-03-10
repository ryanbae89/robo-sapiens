# modules/model_utils.py

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
from peft import LoraConfig

def load_base_model(model_id,
                    quantization_params):
    
    # Convert string to actual dtype
    load_in_4bit = quantization_params["load_in_4bit"]
    bnb_4bit_use_double_quant = quantization_params["bnb_4bit_use_double_quant"]
    bnb_4bit_quant_type = quantization_params["bnb_4bit_quant_type"]
    bnb_4bit_compute_dtype = quantization_params["bnb_4bit_compute_dtype"]
    if bnb_4bit_compute_dtype == "float16":
        compute_dtype = torch.float16
    elif bnb_4bit_compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = None  # fallback, typically float32
            
    # Load bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    device_map = {"": 0}  # single-GPU example

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False
    # tweak as needed
    base_model.config.pretraining_tp = 1
    return base_model

def get_tokenizer(model_id="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = -100
    tokenizer.padding_side = "left"
    return tokenizer

def get_peft_config():
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft_config
