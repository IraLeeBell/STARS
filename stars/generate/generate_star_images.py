import sys
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import math

def generate_star_stack(sx, sy, NN, M, sigma=2.0, RN=10.0, sky_lev=1000.0):
    """
    Generate a set of M astronomical images (M FITS files worth of data).
    One of the stars is made variable (its brightness changes over the images).

    Parameters
    ----------
    sx, sy : int
        Image dimensions in pixels (width, height).
    NN : int
        Number of stars to generate.
    M : int
        Number of images to produce in the stack.
    sigma : float
        RMS width of stars in pixels.
    RN : float
        RMS read noise.
    sky_lev : float
        Sky background counts level.

    Returns
    -------
    images : list of 2D numpy arrays
        A list of M generated images.
    """
    edge = 10

    # Empty image template
    im0 = np.zeros((sx, sy), dtype='float32')

    # Magnitude range
    m_min, m_max = 9.0, 20.27

    # Generate magnitudes
    # Instead of using the original complicated formula for NN, we now trust NN as given.
    cn = np.random.rand(NN)
    m = 2 * np.log(np.exp(0.5*m_min) + cn*(np.exp(0.5*m_max) - np.exp(0.5*m_min)))

    # Star positions, avoiding edges
    x = edge + (np.random.rand(NN) * (sx - 2*edge)).astype('int16')
    y = edge + (np.random.rand(NN) * (sy - 2*edge)).astype('int16')

    # Compute star fluxes
    snr_min = 10.0
    noise = np.sqrt(RN**2 + sky_lev)*np.sqrt(4*math.pi)*sigma
    cts = snr_min * noise * 10**(-0.4*(m - m_max))

    # Signal-to-noise ratio
    snr = cts / np.sqrt(cts + noise**2)

    # Choose a star to make variable
    # We pick one star with a good SNR range
    ii = np.where((snr > 100.) & (snr < 1000.))[0]
    if len(ii) == 0:
        # If no star meets this criterion, just pick one at random
        i0 = np.random.randint(0, NN)
    else:
        i0 = ii[int(np.random.rand()*len(ii))]

    # Frequency of variability
    # We'll pick a frequency related to M, ensuring it's a positive integer range
    # (If M is small, make sure we pick something reasonable)
    low_bound = max(1, int(M/5)-2)
    high_bound = max(2, int(M/5)+2)
    if low_bound >= high_bound:
        low_bound = 1
        high_bound = 3
    f = np.random.randint(low_bound, high_bound)*1.0/M

    # Create an image with just the variable star
    im_s0 = np.zeros((sx, sy), dtype='float32')
    im_s0[x[i0], y[i0]] = cts[i0]

    # Create an image with all the other stars
    im_s = np.zeros((sx, sy), dtype='float32')
    for i in range(NN):
        if i == i0:
            continue
        im_s[x[i], y[i]] += cts[i]

    # Convolve with a Gaussian PSF
    im_s = gaussian_filter(im_s, sigma)
    im_s0 = gaussian_filter(im_s0, sigma)

    # Generate M images
    images = []
    x0 = np.random.rand()*M
    for i in range(M):
        # Vary the variable star sinusoidally
        im_s1 = im_s0*(1 + 0.1*math.sin(2*math.pi*f*(i - x0)))
        # Add noise
        im_i = im0 + (im_s + im_s1 + np.sqrt(RN**2 + im_s + sky_lev)*np.random.randn(sx, sy))
        images.append(im_i.astype('float32'))

    return images

def main():
    args = sys.argv[1:]
    
    if len(args) == 0:
        # No arguments: randomize dimensions and number of stars, default M=10
        sx = np.random.randint(512, 2048)
        sy = np.random.randint(512, 2048)
        NN = np.random.randint(100, 1000)
        M = 10
    elif len(args) == 3:
        # sx sy NN given, but no M specified, default M=10
        try:
            sx = int(args[0])
            sy = int(args[1])
            NN = int(args[2])
        except ValueError:
            print("Arguments must be integers.")
            print("Usage: python generate_star_images.py [sx sy stars [M]]")
            sys.exit(1)
        M = 10
    elif len(args) == 4:
        # sx sy NN M all given
        try:
            sx = int(args[0])
            sy = int(args[1])
            NN = int(args[2])
            M = int(args[3])
        except ValueError:
            print("Arguments must be integers.")
            print("Usage: python generate_star_images.py [sx sy stars [M]]")
            sys.exit(1)
    else:
        print("Usage: python generate_star_images.py [sx sy stars [M]]")
        print("If no arguments are supplied, random values are used.")
        print("If M is not supplied, default is 10.")
        sys.exit(1)
    
    images = generate_star_stack(sx, sy, NN, M)

    # Write each image to a FITS file
    # We'll number them with zero-padding (e.g. image_0001.fits)
    for i, img in enumerate(images, start=1):
        filename = f"synthetic_star_image_{str(i).zfill(4)}.fits"
        fits.writeto(filename, img, overwrite=True)
        print(f"Generated {filename}")

    print(f"Done! Generated {M} images of size {sx}x{sy} with {NN} stars (one variable).")

if __name__ == "__main__":
    main()
