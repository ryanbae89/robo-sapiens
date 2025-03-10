# modules/training_pipeline.py

import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from .model_utils import load_base_model, get_peft_config, get_tokenizer
from .metrics_utils import compute_rouge_metrics
from .data_utils import tokenize_fn

def run_training(
    model_id,
    data,
    format_func,
    output_dir,
    quantization_params,
    num_train_epochs=2,
    learning_rate=2e-4,
    batch_size=4,
    gradient_accumulation_steps=4,
):
    # Load base model + tokenizer
    base_model = load_base_model(model_id=model_id,
                                 quantization_params=quantization_params)
    tokenizer = get_tokenizer(model_id=model_id)
    peft_config = get_peft_config()

    # Map the dataset to tokenized data (format_func is the format_prompt function in prompt_utils)
    def map_fn(sample):
        return tokenize_fn(sample, tokenizer, format_func)
    
    # TODO: map train data to the imported format instruction function
    train_data = data["train"].map(map_fn)
    print(train_data[0].keys())
    
    # Optionally, also map test_data or validation data

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=base_model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        # Provide a custom compute_metrics function if you have test/val data
        compute_metrics=lambda eval_preds: compute_rouge_metrics(eval_preds, tokenizer),
        dataset_text_field="text"  # specify if you want SFTTrainer to handle text field
    )

    trainer.train()
    return trainer
