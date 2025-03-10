# modules/prompt_utils.py

def format_instructions(group_description, responder_name, speakers):
    return f"""Your name is {responder_name}. You are in a group chat with: {speakers}. {group_description} You are encouraged to curse, say generally inappropriate and hilarious things, or be very rude. Everything is in good fun, so remember to joke and laugh, and be funny, but also answer all questions clearly. You will be presented with the most recent messages in the group chat. Write a response to the conversation as {responder_name}."""

def format_prompt(sample):
	return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['input']}

### Response:"""