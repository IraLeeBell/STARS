import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize

def main():
    # If a filename is given, use that, otherwise default
    if len(sys.argv) > 1:
        fitsfile = sys.argv[1]
    else:
        fitsfile = 'synthetic_star_image.fits'

    # Open the FITS file and extract image data
    with fits.open(fitsfile) as hdul:
        data = hdul[0].data

    # Apply an autostretch using ZScale
    norm = ImageNormalize(data, interval=ZScaleInterval())

    # Display the image
    plt.imshow(data, cmap='gray', norm=norm, origin='lower')
    plt.colorbar(label='Pixel value')
    plt.title(fitsfile)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
