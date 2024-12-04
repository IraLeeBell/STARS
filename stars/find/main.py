import numpy as np
from imsim import imsim
from scipy.ndimage import gaussian_filter, maximum_filter
    


def find_stars(image, window_size=3):
    smoothed_image = gaussian_filter(image, sigma=2.0)
    
    
    threshold = np.mean(smoothed_image) + 0.70 * np.std(smoothed_image)
    star_rows, star_cols = np.where(smoothed_image > threshold)
    confirmed_rows, confirmed_cols = [], []
    
    
    for row, col in zip(star_rows, star_cols):
        if row - window_size < 0 or row + window_size >= image.shape[0] or col - window_size < 0 or col + window_size >= image.shape[1]:
            continue
        
        local_area = image[row - window_size : row + window_size + 1, col - window_size : col + window_size + 1]
        
        if image[row, col] == np.max(local_area):
            confirmed_rows.append(row)
            confirmed_cols.append(col)
    
    return np.array(confirmed_rows), np.array(confirmed_cols)
     
if __name__ == '__main__':
    
    image = imsim()
    i,j = find_stars(image)
    
    print ("I found {:d} stars!".format(len(i)))
