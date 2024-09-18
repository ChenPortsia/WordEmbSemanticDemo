from typing import List, Tuple

def validate_input(x_space: Tuple[List[str], List[str]], y_space: Tuple[List[str], List[str]], 
                   groups: List[List[str]], operation: str, target_group: str, extra_word: str) -> Tuple[bool, str]:
    """Validate user input."""
    if not all(x_space) or not all(y_space):
        return False, "X and Y space words cannot be empty."
    
    if not all(groups):
        return False, "All groups must contain at least one word."
    
    if len(groups) > 3:
        return False, "Maximum 3 groups are allowed."
    
    if operation and operation not in ['add', 'subtract', 'multiply', 'divide', 'average']:
        return False, "Invalid operation."
    
    if target_group not in ['all', 'group_1', 'group_2', 'group_3']:
        return False, "Invalid target group."
    
    if operation and not extra_word:
        return False, "Extra word is required when an operation is specified."
    
    return True, ""

def parse_input(input_str: str) -> List[str]:
    """Parse comma-separated input string into a list of words."""
    return [word.strip() for word in input_str.split(',') if word.strip()]