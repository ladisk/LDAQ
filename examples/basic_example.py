"""
This is a basic use case of the SDyPy template project.
"""

import sdypy_template_project as iep
import numpy as np
import matplotlib.pyplot as plt

video = np.load('examples/speckle.npy', mmap_mode='r')
results = iep.get_displacements(video, point=[5, 5], roi_size=[7, 7])

plt.figure()
plt.plot(results[0], label='x')
plt.plot(results[1], label='y')
plt.xlabel('frame [/]')
plt.ylabel('displacement [pixel]')
plt.legend()
plt.show()