# modules/data_utils.py

import os
import random
import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_split_data(json_file_path, sample_size=4000, test_ratio=0.1):
    with open(json_file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle or sample if needed
    if sample_size:
        data = random.sample(data, sample_size)

    dataset = Dataset.from_list(data)
    dataset_split = dataset.train_test_split(test_size=test_ratio)
    return dataset_split

def tokenize_fn(example, tokenizer, format_func):
    # This is where you'd apply your custom "instruction" format:
    text = format_func(example)
    tokenized = tokenizer(text)
    return {"text": tokenizer.decode(tokenized["input_ids"])}
