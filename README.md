# robo-sapiens

**Robo-Friends** is a project that LoRA fine-tunes LLMs on group chat data to mimic the personalities and writing styles of specific friend groups. It includes:

1. **Data Processing** (from raw chat exports to training-ready data)  
2. **LLM Training** (LoRA fine-tuning on top of a base model)  
3. **Inference** (generating conversation responses using the fine-tuned adapters)  
4. **Orchestrator** (end-to-end pipeline combining data processing, training, and inference)

## Directory Structure

### High-Level Overview

- **`data_processing/`**  
  Code to gather and clean raw chat logs (WhatsApp, etc.) and produce structured JSON for training.

- **`training/`**  
  Scripts and modules to fine-tune a base LLM (e.g., Llama-2) using **LoRA** adapters.

- **`inference/`**  
  Load the fine-tuned model + LoRA adapters to generate responses for new inputs.

- **`orchestrator/`**  
  A top-level pipeline script to run data processing → training → inference in one go.

- **`app/`**  
  A React-based chat-like frontend (iMessenger clone) for interactive demonstration.

---

## 1. Setup & Installation

1. **Clone the Repository**  
```bash
   git clone https://github.com/your-username/robo-sapiens.git
   cd robo-sapiens
```

2. **Install Python Dependencies**

* It’s recommended to use a virtual environment (conda, venv, etc.) in Linux. Currently recommend using Python 3.11.

```
conda create -n robo-sapiens python=3.11
conda activate robo-sapiens
pip install -r requirements.txt
```

3. ** Install Node Dependencies (for the React app)**

```
cd app
yarn install    # or npm install
```

4. **(Optional) Docker Build (NOT READY YET)** 

If you have a Dockerfile for your entire application, run:

```
docker build -t robo-sapiens:latest
```

## 2. Data Processing

Goal: Convert raw WhatsApp `.txt` (or other chat formats) into training-ready JSON.

* Key Script: `data_processing/scripts/whatsapp_process_pipeline.py`
* Example Usage (if using command line):

```
python -m data_processing.scripts.whatsapp_process_pipeline \
  --config data_processing/configs/whatsapp_data_config.yaml
```

* Function approach:

```
from data_processing.scripts.whatsapp_process_pipeline import run_whatsapp_pipeline

run_whatsapp_pipeline(
    config="data_processing/configs/whatsapp_data_config.yaml",
    raw_input_file="path/to/chat_export.txt",
    output_dir="path/to/output_dir",
    chat_info={"group_description": "..."}
)
```

* Output: Typically `messages_processed.json` with conversation segments (`instruction`, `input`, `output`, `responder_name`, etc.).

## 3. Training

Goal: Fine-tune a base model (e.g., Llama-2) using LoRA to learn the style/voice of the chat group.

* Key Script: `training/scripts/train_llm.py`
* Config: `training/configs/training_config.yaml`
* Example Command:

```
python -m training.scripts.train --config training/configs/training_config.yaml
```

* Result: LoRA adapter weights saved to `"output_dir"`. The base model is not duplicated; only the small LoRA checkpoint is stored.

## 4. Inference

Goal: Load the base model + LoRA adapter and generate conversation responses to a new prompt.

* Key Script: `inference/scripts/generate_test_conversation.py`
* Configs: 
    * `model_config.yaml` (which includes base_model_id, lora_path, quantization),
    * `generation_config.yaml` (for temperature, max_new_tokens, etc.).
* Example Usage:

```
python -m inference.scripts.generate_test_conversation \
  --model-config inference/configs/model_config.yaml \
  --generation-config inference/configs/generation_config.yaml
```

* Function approach (`run_inference_test`) used in your orchestrator:

```
from inference.scripts.generate_test_conversation import run_inference_test

convo = run_inference_test(
    model_config_path="inference/configs/model_config.yaml",
    generation_config_path="inference/configs/generation_config.yaml",
    context="<Speaker Name>: context text here..."
)
print(convo)
```

## 5. Orchestrator

Goal: Run an end-to-end pipeline from raw data → training → inference.

* Script: `orchestrator/master_orchestrator.py`
* Config: `orchestrator/configs/full_orchestrator_config.yaml`
* Example:

```
python -m orchestrator.master_orchestrator
```

* What It Does:
    1. Data Processing: Reads raw chat logs, outputs processed JSON.
    2. Training: Fine-tunes LoRA adapter using the processed data.
    3. Inference: Uses the newly trained adapter to generate a test conversation.
    4. Outputs: Final config dump, logs, LoRA model artifacts, etc.

Sample `full_orchestrator_config.yaml`:

```
chat_info:
  group_members: ["John Doe", "Mary Jane", ...]
  group_description: "Short group description here..."
  phone_number_mapping:
    "1234567890": "John Doe"
    "2345678901": "Mary Jane"
    ...

data_processing_pipeline_params:
  config: "data_processing/configs/whatsapp_data_config.yaml"
  raw_input_file: "path/to/chat_export.txt"
  output_dir: "path/to/output_dir"

training_pipeline_params:
  config: "llm_training/configs/training_config.yaml"
  output_dir: "path/to/output_dir"

inference_pipeline_params:
  model_config: "inference/configs/model_config.yaml"
  generation_config: "inference/configs/generation_config.yaml"
  test_speaker: "<Speaker Name>"
  test_prompt: "your test prompt here..."
```

## 6. Running the React App (Optional)

If you have the React UI that mocks an iMessenger-like interface:

Install dependencies in app/ folder:

```
cd app
yarn install
yarn start
```

The app calls an inference API endpoint to get chat responses. You can run or modify a start_server.py in your inference/ folder to serve model predictions.

## 7. Troubleshooting

* ModuleNotFoundError

    * Make sure you’re running from the project root with `python -m ...`, and you have `__init__.py` in each folder.
    * Alternatively, install the repo in editable mode: `pip install -e`.
    
* 4-bit Quantization

    * If using `bitsandbytes` for 4-bit, ensure you pass the correct `BitsAndBytesConfig`.
    * You may need sufficient GPU memory or partial CPU offloading if the model is large.

* Assertion Errors in bitsandbytes

    * Usually indicates the model’s 4-bit layers aren’t fully moved to GPU. Call `model.to("cuda")` or set `device_map={"":0}`.

* Memory Issues

    * Large models like Llama-2 can require significant GPU memory. Use smaller batch sizes, gradient accumulation, or 4-bit quantization to reduce usage.

## 8. Future Plans

* Full API integration with WhatsApp
* Integration with additional chat data sources (Instagram, iMessage, Discord, etc).
* Write unittests for each module
* LoRA on Lllam-3 series models and hyperparmeter tuning
* Evaluation metric integration into test module
* App UI / Chat integrated to dynamically select different LoRA adapters.
* Experiment with single LoRA adapter weights per person, instead of per conversation

## 9. Contributing

* Fork the repo and create a new branch for your feature or bugfix.
* Submit a pull request with a clear description.
* Ensure you run the test suites in `data_processing/tests/`, `training/tests/`, `inference/tests/` (when they are available!).

## 10. License

Working on this part...

## 11. Contact
For questions or collaboration:

* Creator: Ryan Bae

* Email: ryanbae89@gmail.com

* [LinkedIn](https://www.linkedin.com/in/ryanbae89/)

Feel free to open an issue or pull request if you find a bug or have a suggestion!