# modules/prompt_utils.py

def format_prompt(sample):
	return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['input']}

### Response:
{sample["responder_name"]}: {sample['output']}</s>"""