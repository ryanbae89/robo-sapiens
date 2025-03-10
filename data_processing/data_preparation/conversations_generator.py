# data_preparation/conversation_splitter.py
import re
from datetime import datetime, timedelta


def generate_conversations(messages, training_data_params):
    """
    """
    gap_minutes = training_data_params["conversation_gap_minutes"]
    
    conversations = []
    current_convo = []
    prev_dt = None
    
    for message in messages:
        dt = message["timestamp"]
        if dt is None:
            # skip or handle lines that don't parse
            continue
        # If no previous message or the gap is large -> new conversation
        if prev_dt is None or (dt - prev_dt) > timedelta(minutes=gap_minutes):
            if current_convo:
                conversations.append(current_convo)
            current_convo = []
        
        current_convo.append(message)
        prev_dt = dt
    
    # Append last convo if not empty
    if current_convo:
        conversations.append(current_convo)
    
    print(f"Split into {len(conversations)} conversations.")
    return conversations