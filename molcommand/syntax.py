from unitymol_copilot import scene_context
import json

# Ensure valid commands
def validate_command(command):
    valid_commands = ["load", "show", "color_selection", "select"]
    return command.lower() in valid_commands

def process_command(command):
    if validate_command(command):
        # Proceed with valid commands (e.g., sending to UnityMolX)
        result = run_command(command)
        return result
    else:
        return "Invalid command."

