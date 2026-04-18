# utils.py
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button

# The 8 specific actions we care about
# We'll map them 0 to 7
ACTIONS = {
    0: 'w',
    1: 'a',
    2: 's',
    3: 'd',
    4: 'space',
    5: 'shift',
    6: 'click_left',
    7: 'click_right'
}

# Reverse mapping for logging
ACTION_TO_INDEX = {v: k for k, v in ACTIONS.items()}

def pynput_key_to_action(key):
    """ Converts a pynput key object to our internal string action, if it matches. """
    if isinstance(key, KeyCode):
        char = key.char
        if char is not None:
            char = char.lower()
            if char in ['w', 'a', 's', 'd']:
                return char
    elif isinstance(key, Key):
        if key == Key.space:
            return 'space'
        elif key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            return 'shift'
    return None

def pynput_mouse_to_action(button):
    """ Converts a pynput mouse Button to our internal string action. """
    if button == Button.left:
        return 'click_left'
    elif button == Button.right:
        return 'click_right'
    return None

def action_to_pynput(action_str):
    """ Converts our action string back to pynput Key or Button for playback. """
    if action_str in ['w', 'a', 's', 'd']:
        return KeyCode.from_char(action_str)
    elif action_str == 'space':
        return Key.space
    elif action_str == 'shift':
        return Key.shift
    elif action_str == 'click_left':
        return Button.left
    elif action_str == 'click_right':
        return Button.right
    return None
