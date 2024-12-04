import numpy as np
from imsim import imsim
from scipy.ndimage import gaussian_filter

        

def measure_stars(image, x, y):
    
    smoothed_image = gaussian_filter(image, sigma=2)
    
    flux = np.zeros(len(x))
    
    for idx in range(len(x)):
        i, j = x[idx], y[idx]
        flux[idx] = np.sum(smoothed_image[i-1:i+2, j-1:j+2])

    return flux
    
    
    
if __name__ == '__main__':
    
    image,i,j = imsim()
    flux = measure_stars(image,i,j)
    
    print ("I measured {:d} stars!".format(len(flux)))
