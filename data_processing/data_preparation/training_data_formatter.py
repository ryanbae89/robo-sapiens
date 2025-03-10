# data_preparation/training_data_formatter.py

def build_training_samples(
    conversations,
    params,
    group_description,
    group_members
):
    """
    Given a list of conversations (each a list of messages),
    produce a list of training sample dicts:
    {
      "instruction": ...,
      "input": "Conversation excerpt so far",
      "output": "The next speaker's message",
      "responder_name": "Who wrote the next message"
    }
    """
    max_context_size = params["max_conversation_size"]
    training_samples = []

    for convo in conversations:
        # e.g. convo is a list of message dict: {"timestamp", "speaker", "message"}
        # We'll walk through each message after the first
        for i in range(1, len(convo)):
            # up to i-1 is context, i is the next message to predict
            context_slice = convo[max(0, i - max_context_size) : i]
            next_msg = convo[i]

            # Build the "input" by concatenating messages in context
            context_text = []
            for msg in context_slice:
                context_text.append(f"{msg['sender']}: {msg['message']}")

            input_text = "\n".join(context_text)
            instruction_text = generate_prompt(group_description,
                                               next_msg["sender"],
                                               group_members)
            training_samples.append(
                {
                    "instruction": instruction_text,
                    "input": input_text,
                    "output": next_msg["message"],
                    "responder_name": next_msg["sender"],
                }
            )

    return training_samples

def generate_prompt(group_description, responder_name, speakers):
    return f"""Your name is {responder_name}. You are in a group chat with: {speakers}. {group_description} You are encouraged to curse, say generally inappropriate and hilarious things, or be very rude. Everything is in good fun, so remember to joke and laugh, and be funny, but also answer all questions clearly. You will be presented with the most recent messages in the group chat. Write a response to the conversation as {responder_name}."""
