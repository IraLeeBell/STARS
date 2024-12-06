import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
    # Find all FITS files matching the pattern
    files = sorted(glob.glob("synthetic_star_image_*.fits"))
    if len(files) == 0:
        print("No images found. Make sure your FITS files are named 'synthetic_star_image_XXXX.fits'.")
        return

    # Load all images into a 3D array: shape (M, sx, sy)
    images = []
    for f in files:
        with fits.open(f) as hdul:
            data = hdul[0].data
            images.append(data)
    images = np.array(images)  # shape: (M, sx, sy)

    M, sx, sy = images.shape
    print(f"Loaded {M} images of size {sx}x{sy}")

    # Find the pixel with the largest standard deviation across time
    std_map = np.std(images, axis=0)
    max_coord = np.unravel_index(np.argmax(std_map), std_map.shape)
    star_x, star_y = max_coord

    # Extract the flux values for this star across all images
    flux_values = images[:, star_x, star_y]

    # Calculate flux variation stats
    flux_min = np.min(flux_values)
    flux_max = np.max(flux_values)
    flux_range = flux_max - flux_min
    flux_mean = np.mean(flux_values)

    # Choose one image to display (e.g., the first one)
    chosen_image_idx = 0
    chosen_data = images[chosen_image_idx]

    # Plot the chosen image
    fig, ax = plt.subplots(figsize=(6,6))
    norm_min, norm_max = np.percentile(chosen_data, [5, 95])
    ax.imshow(chosen_data, cmap='gray', origin='lower', vmin=norm_min, vmax=norm_max)

    # Draw a red circle around the variable star
    circ = Circle((star_y, star_x), radius=5, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circ)

    # Construct a legend with relevant info
    info_text = (f"Variable Star:\n"
                 f"Coordinates (x,y): ({star_x}, {star_y})\n"
                 f"Flux range: {flux_min:.1f} to {flux_max:.1f}\n"
                 f"Flux variation: ±{flux_range/2:.1f}")

    # Place a text box with these details
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # Inset plot for flux variation
    inset_ax = inset_axes(ax, width="40%", height="30%", loc='lower left', borderpad=1)
    inset_ax.plot(range(1, M+1), flux_values, marker='o', color='red', linestyle='-')
    inset_ax.set_title("Flux Variation", fontsize=8)
    inset_ax.set_xlabel("Image #", fontsize=8)
    inset_ax.set_ylabel("Flux", fontsize=8)
    inset_ax.tick_params(labelsize=8)

    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig("variable_star_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
