import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits

def generate_star_image(sx, sy, N):
    # Fixed magnitude range
    m_min, m_max = 9.0, 20.27
    
    # Start with read-noise: Gaussian noise with mean=0 and sigma=10
    im = 10.0 * np.random.randn(sx, sy)
    
    # Generate random magnitudes for the N stars
    # cn is a uniform random sample used to derive magnitudes
    cn = np.random.rand(N)
    m = 2 * np.log(np.exp(0.5*m_min) + cn * (np.exp(0.5*m_max)-np.exp(0.5*m_min)))

    # Random star positions, avoiding edges (10-pixel border)
    x = 10 + (np.random.rand(N) * (sx-20)).astype('int16')
    y = 10 + (np.random.rand(N) * (sy-20)).astype('int16')

    # Compute star counts (flux) based on magnitude
    cts = 10 * np.sqrt(1.1e3) * 10**(-0.4*(m-m_max)) * np.sqrt(4*np.pi)*2

    # Place the stars into an empty image
    im_s = np.zeros((sx, sy), dtype='float32')
    for i in range(N):
        im_s[x[i], y[i]] += cts[i]

    # Apply a Gaussian PSF and add Poisson noise (sqrt(im_s+1e3)) scaled random noise
    im_starred = gaussian_filter(im_s, 2.0) + np.sqrt(im_s + 1.0e3)*np.random.randn(sx, sy)
    
    # Combine with the initial read-noise image
    final_image = im + im_starred
    
    return final_image

def main():
    # Parse command line arguments
    args = sys.argv[1:]
    
    if len(args) == 0:
        # No arguments supplied: randomize dimensions and number of stars
        sx = np.random.randint(512, 2048)
        sy = np.random.randint(512, 2048)
        N = np.random.randint(100, 1000)
    elif len(args) == 3:
        # Use provided arguments
        try:
            sx = int(args[0])
            sy = int(args[1])
            N = int(args[2])
        except ValueError:
            print("Arguments must be integers.")
            print("Usage: python generate_star_image.py [sx sy N]")
            sys.exit(1)
    else:
        # Incorrect number of arguments
        print("Usage: python generate_star_image.py [sx sy N]")
        print("If no arguments are supplied, random values are used.")
        sys.exit(1)

    # Generate the image
    image_data = generate_star_image(sx, sy, N)

    # Write to a FITS file
    # You can change the output filename as desired
    output_filename = "synthetic_star_image.fits"
    fits.writeto(output_filename, image_data, overwrite=True)

    print(f"Image generated: {output_filename}")
    print(f"Dimensions: {sx}x{sy}, Number of stars: {N}")

if __name__ == "__main__":
    main()
