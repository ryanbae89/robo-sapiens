# data_fetchers/whatsapp_fetcher.py

import os
import re
from datetime import datetime

from data_processing.data_cleaning.cleaning_utils import replace_phone_numbers

def load_and_parse_whatsapp_txt(file_path, params, phone_number_mapping):
    """
    Placeholder function to load raw lines from a local .txt file
    that was exported from WhatsApp.
    """
    remove_non_text_messages = params["remove_non_text_messages"]
    remove_system_messages = params["remove_system_messages"]
    
    message_pattern = re.compile(r'^\[(.*?)\]\s+(.*?):\s+(.*)$')
    dirchars_pattern = re.compile(r'[\u200e\u200f\u202a\u202b\u202c\u202d\u202e]')
    omitted_pattern = re.compile(r'^\w+\s+omitted$', re.IGNORECASE)
    link_only_pattern = re.compile(r'^(?:https?://\S+)(?:\s+(?:https?://\S+))*$')
    data = []
    n_skipped = 0
    
    valid_speakers = list(phone_number_mapping.values())
    
    # if file does not exists, throw error
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # open and parse the file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if dirchars_pattern.search(line):
                n_skipped += 1
                continue
            # remove trailing newline
            line = line.rstrip('\n') 
            
            # Skip completely empty lines if desired
            if not line.strip():
                continue
            
            # check if main message pattern is a match
            match = message_pattern.match(line)
            if match:
                
                # If this line starts with a [timestamp], it's a new message.
                timestamp_str = match.group(1)
                sender = match.group(2)
                message = match.group(3)
                # datetime formatting
                dt_format = '%m/%d/%y, %I:%M:%S %p'
                parsed_dt = datetime.strptime(timestamp_str, dt_format)
                
                if remove_non_text_messages:
                    # Skip if it's an "omitted" placeholder (e.g., "image omitted", "GIF omitted")
                    if omitted_pattern.match(message):
                        n_skipped += 1
                        continue

                    # Skip if the entire message is only link(s)
                    if link_only_pattern.match(message):
                        n_skipped += 1
                        continue
                
                if remove_system_messages:
                    # skip if sender is not in valid speakers
                    if sender not in valid_speakers:
                        n_skipped += 1
                        continue
                
                # sub message if @phone number with name
                message = replace_phone_numbers(message, phone_number_mapping)
                
                # append to data
                data.append({
                    'timestamp': parsed_dt,
                    'sender': sender,
                    'message': message
                })

            else:
                # If line doesn't match, assume it's a continuation of the previous message
                if data:
                    # Append this line to the last message in the list
                    data[-1]['message'] += '\n' + line
                else:
                    # If there's no previous entry to attach to,
                    # you might decide to create a new entry or skip
                    # For safety, let's just skip or log it:
                    #   print(f"Orphaned line (no preceding message): {line}")
                    pass
    print(f"{len(data)} messages parsed. {n_skipped} messages skipped.")
    return data