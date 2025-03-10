# data_cleaning/cleaning_utils.py

import re

def replace_phone_numbers(message, phone_numbers_mapping):
    """
    Replace phone numbers with user-friendly names using a dict map.
    phone_to_name_map = {"15551234567": "John Doe", ...}
    
    """
    for number, name in phone_numbers_mapping.items():
        old_str = f"@{number}"
        new_str = f"@{name}" 
        message = message.replace(old_str, new_str)
    return message
