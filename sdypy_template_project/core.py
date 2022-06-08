import numpy as np
from tqdm import tqdm

# get_displacements
from scipy.optimize import least_squares
from scipy.interpolate import interp2d


def cost_function(p, x, y, f, g, W):
    """
    The cost function to minimize in the optimization process. The element-wise
    difference between the reference image f(x, y) and a tansformed current
    image g(W(x, y)).
    
    Args:
        p (array of shape (n,)): The geometric transform parameters to
            optimize.
        x (array of shape (w,)): The x-coordinates of reference image 
            pixel grid.
        y (array of shape (h,)): The y-coordinates of reference image 
            pixel grid.
        f (array of shape (H, W)): The reference image.
        g (array of shape (H, W)): The current image.
        W (callable): The geometric transform function with the signature 
            `W(x, y, p, g)`, that evaluates and returns the current image
            values at transformed coordinates for the parameters `p`.

    Returns:
        residual (array of shape (w*h,)): Vector of residuals.
    """
    x_mesh, y_mesh = np.meshgrid(x, y)
    return (f[y_mesh, x_mesh] - W(x, y, p, g)).flatten()


def translation(x, y, p, g):
    """
    Geometric transform function of simple translation. Interpolates the image
    g(x, y) using cubic B-splineinterpolation and evaluates its pixel values
    at g(y+p[1], x+p[0]).
    
    Args:
        x (array of shape (w,)): The x-coordinates of reference image 
            pixel grid.
        y (array of shape (h,)): The y-coordinates of reference image 
            pixel grid.
        p (array of shape (2,)): The translation parameters [dx, dy].
        g (array of shape (H, W)): The image to be evaluated.
    
    Returns:
        residual (array of shape (h, w)): The transformed image.
    """
    dx, dy = p
    h, w = g.shape
    spl = interp2d(np.arange(w), np.arange(h), g, kind='cubic')
    return spl(x + dx, y + dy)


def roi_xy(point, roi_size):
    """
    Get grid coordinates of the selected region of interest.
    
    Args:
        point (array of shape (2,)): The position of the point to analyze,
            [X, Y].
        roi_size (array of shape (2,)): The region-of-interest dimensions,
            [W, H]. The ROI dimensions must be odd numbers!
    Returns:
        x (array of shape (W,)): The x coordinates of the ROI point grid.
        y (array of shape (H,)): The y coordinates of the ROI point grid.
    """

    W, H = roi_size

    # ROI domensions must be odd!
    if not W % 2:
        W += 1
    if not H % 2:
        H += 1

    x_start = point[0] - W//2
    y_start = point[1] - H//2
    x = np.arange(x_start, x_start + W)
    y = np.arange(y_start, y_start + H)
    return x, y


def get_displacements(images, point, roi_size):
    """
    Calculate displacements of a selected point in the image. Uses the
    Lucas-Kanade image alignment algorithm for simple translations.
    
    Args:
        images (array of shape (N, h, w)): Image data array.
        point (array of shape (2,)): The position of the point to analyze,
            [X, Y].
        roi_size (array of shape (2,)): The region-of-interest dimensions,
            [W, H]. The ROI dimensions must be odd numbers!
    
    Returns:
        d (array of shape (2, N)): The image-identified translations [dx, dy]
            (in pixels).
    """
    f = images[0]
    x, y = roi_xy(point, roi_size)

    d = np.zeros((2, images.shape[0]))
    for i, g in enumerate(tqdm(images)):
        p_opt = least_squares(cost_function, x0=(0, 0), args=(x, y, f, g, translation))
        d[:, i] = p_opt.x
    return d