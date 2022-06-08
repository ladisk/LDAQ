'''
The code inside `__main__.py` runs when this module is run from the command line with:
    
    python -m sdypy_template_project
'''

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt
from matplotlib import animation
from pyMRAW import load_video

from .visualize import animate_video

import warnings
warnings.filterwarnings("ignore")

# Open file selection window to select the MRAW video file
window = Tk()
filename = askopenfilename(parent=window, title='Select the .cih file', filetypes=[
    ("Photron cih file", "*.cih"), ("Photron cihx file", "*.cihx")])

# Close the Tk window
window.destroy()

# Load the video
images, cih = load_video(filename)

# Show the animation
ani = animate_video(images, fps=30, bit_depth=cih['EffectiveBit Depth'])
plt.show()