import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches


def show_reference_image(images, points=[], roi_size=None, bit_depth=12):
    """
    Display the reference image of a MRAW video sequence. together with
    selected points  and region-of-interest borders.
    
    Args:
        images (array of shape (n, h, w)): The image sequence.
        points (array of shape (N, 2)): The points to draw on top of the image.
        roi_size (array of shape (2,)): The height and width of the ROI.
        bit_depth (int): Effective bit depth of the captured images.

    Returns:
        fig: Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.imshow(images[0], cmap='gray', vmin=0, vmax=2**bit_depth)

    if points:
        ax.scatter(np.array(points)[:, 1], np.array(points)[:, 0], marker='.', color='r')
        
        if roi_size is not None:
            for point in np.array(points):
                roi_border = patches.Rectangle((point - roi_size//2)[::-1], roi_size[1], roi_size[0],
                                               linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(roi_border)

    return fig


def animate_video(images, fps=30, bit_depth=12):
    """
    A function that animates a sequence of grayscale images using matplotlib.
    
    Args:
        images (array of shape (n, h, w)): The image sequence.
        fps (int): The playback framerate.
        bit_depth (int): Effective bit depth of the captured images.

    Returns:
        ani: Matplotlib animation object.
    """
    fig = plt.figure()
    
    # display data for first image
    im = plt.imshow(images[0], cmap='gray', vmin=0, vmax=2**bit_depth, animated=True) 

    def updatefig(i):
        im.set_array(images[i])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, blit=True, frames=images.shape[0], 
        interval=1000/fps)
    return ani
