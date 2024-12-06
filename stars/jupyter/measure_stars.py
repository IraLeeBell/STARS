import numpy as np
from scipy.ndimage import gaussian_filter

def measure_stars(image, x, y):
    smoothed_image = gaussian_filter(image, sigma=2)
    flux = np.zeros(len(x))
    
    for idx in range(len(x)):
        j, i = x[idx], y[idx]  # x is column (j), y is row (i)
        flux[idx] = np.sum(smoothed_image[i-1:i+2, j-1:j+2])
    
    return flux

