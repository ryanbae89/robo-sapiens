# scripts/whatsapp_process_pipeline.py

import os
import json
import argparse
import yaml

from data_processing.data_fetchers.whatsapp_fetcher import load_and_parse_whatsapp_txt
from data_processing.data_preparation.conversations_generator import generate_conversations
from data_processing.data_preparation.training_data_formatter import build_training_samples


def run_whatsapp_pipeline(config: str,
                          raw_input_file: str,
                          output_dir: str,
                          chat_info: dict):
    """
    Functional method to run the WhatsApp processing pipeline given a YAML config file path.
    Steps:
      1) Load config from YAML
      2) Load and parse raw WhatsApp text messages
      3) Generate conversations
      4) Build training samples
      5) Write final JSON output
    Returns the path to the output JSON file.
    """

    # --- Load config from YAML ---
    print(f"[run_whatsapp_pipeline] Loading config file: {config}")
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    
    # --- Extract config fields ---
    data_parsing_params = config["data_parsing_params"]
    training_data_params = config["training_data_params"]
    group_members = chat_info["group_members"]
    group_description = chat_info["group_description"]
    phone_number_mapping = chat_info["phone_number_mapping"]

    # 1) Load and parse raw text messages
    print(f"[run_whatsapp_pipeline] Loading raw WhatsApp text from: {raw_input_file}")
    parsed_lines = load_and_parse_whatsapp_txt(raw_input_file, 
                                               data_parsing_params,
                                               phone_number_mapping)

    # 2) Generate conversations
    print("[run_whatsapp_pipeline] Generating conversations...")
    conversations = generate_conversations(parsed_lines, 
                                           training_data_params)

    # 3) Build training samples
    print("[run_whatsapp_pipeline] Building training samples...")
    training_data = build_training_samples(conversations, 
                                           training_data_params,
                                           group_description,
                                           group_members)

    # 4) Output JSON
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "messages_processed.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"[run_whatsapp_pipeline] Final training data saved to {out_file}")
    return out_file


def main():
    """
    CLI entry point. Parses inputs and calls run_whatsapp_pipeline.
    """
    parser = argparse.ArgumentParser(description="Process WhatsApp messages data.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--raw_input_file", type=str, required=True, help="Path raw chat .txt file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to output processed training data.")
    args = parser.parse_args()

    run_whatsapp_pipeline(args.config)


if __name__ == "__main__":
    main()
