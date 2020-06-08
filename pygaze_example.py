from pygaze.canvas import canvas
from pygaze.keyboard import keyboard
# Create a keyboard and a canvas object
my_keyboard = keyboard(exp, timeout=0)
my_canvas = canvas(exp)
# Loop ...
while True:
    # ... until space is pressed
    key, timestamp = my_keyboard.get_key()
    if key == 'space':
        break
    # Get gaze position from pygaze ...
    x, y = exp.pygaze_eyetracker.sample()
    # ... and draw a gaze-contingent fixation dot!
    my_canvas.clear()
    my_canvas.fixdot(x, y)
    my_canvas.show()