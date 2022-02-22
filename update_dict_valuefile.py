import json
"""
This function updates the value of a dictionary set in a file"""

def update_value_dict(file_name, prev_key, new_val):
    with open(file_name, 'r') as f:
        data = json.load(f)
        # current_value = data[prev_key]
    data[prev_key] = new_val

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)
    