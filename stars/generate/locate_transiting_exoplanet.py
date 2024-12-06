import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
    # Load star info
    info_file = "exoplanet_star_info.npz"
    try:
        data = np.load(info_file, allow_pickle=True)
        star_coords = data['star_coords']
        transit_star_idx = data['transit_star_idx'].item()
        ref_star_idx = data['ref_star_idx'].item()
        sx = data['sx'].item()
        sy = data['sy'].item()
        M = data['M'].item()
        NN = data['NN'].item()
    except FileNotFoundError:
        print(f"File {info_file} not found. Make sure you ran the generation script first.")
        return

    files = sorted(glob.glob("images/exoplanet_image_*.fits"))
    if len(files) == 0:
        print("No images found. Make sure your FITS files are named 'exoplanet_image_XXXX.fits'.")
        return

    # Load images
    images = []
    for f in files:
        with fits.open(f) as hdul:
            images.append(hdul[0].data)
    images = np.array(images)

    # Extract flux for transit star and reference star
    t_x, t_y = star_coords[transit_star_idx]
    r_x, r_y = star_coords[ref_star_idx]

    transit_flux = images[:, t_x, t_y]
    ref_flux = images[:, r_x, r_y]
    # Normalize by reference flux
    ref_flux_safe = np.where(ref_flux == 0, 1, ref_flux)
    rel_flux = transit_flux / ref_flux_safe
    print("Any NaN in rel_flux?", np.isnan(rel_flux).any())
    print("Any Inf in rel_flux?", np.isinf(rel_flux).any())
    print("rel_flux min, max:", np.nanmin(rel_flux), np.nanmax(rel_flux))


    # Compute the flux drop
    flux_min = np.min(rel_flux)
    flux_max = np.max(rel_flux)
    flux_drop_percent = (1 - flux_min) * 100

    print("ref_flux min:", np.min(ref_flux), "ref_flux max:", np.max(ref_flux))
    print("Any NaN in images?", np.isnan(images).any())


    # Choose one image to display
    chosen_image_idx = 0
    chosen_data = images[chosen_image_idx]

    # Determine a good placement for the text and inset
    # We'll decide based on which quadrant the star is in.
    # Quadrants (relative to center):
    # top-left, top-right, bottom-left, bottom-right
    # We'll place annotation and inset in the diagonally opposite corner.
    half_sx = sx / 2
    half_sy = sy / 2

    if t_x < half_sx and t_y < half_sy:
        # Star in top-left quadrant -> place text bottom-right
        text_x, text_y = 0.95, 0.05
        text_ha, text_va = 'right', 'bottom'
        inset_loc = 'upper right'
    elif t_x < half_sx and t_y >= half_sy:
        # Star in top-right quadrant -> place text bottom-left
        text_x, text_y = 0.05, 0.05
        text_ha, text_va = 'left', 'bottom'
        inset_loc = 'upper left'
    elif t_x >= half_sx and t_y < half_sy:
        # Star in bottom-left quadrant -> place text top-right
        text_x, text_y = 0.95, 0.95
        text_ha, text_va = 'right', 'top'
        inset_loc = 'lower right'
    else:
        # Star in bottom-right quadrant -> place text top-left
        text_x, text_y = 0.05, 0.95
        text_ha, text_va = 'left', 'top'
        inset_loc = 'lower left'

    fig, ax = plt.subplots(figsize=(6,6))
    norm_min, norm_max = np.percentile(chosen_data, [5, 95])
    ax.imshow(chosen_data, cmap='gray', origin='lower', vmin=norm_min, vmax=norm_max)

    circ = Circle((t_y, t_x), radius=5, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circ)

    info_text = (f"Transiting Star:\n"
                 f"Coordinates (x,y): ({t_x}, {t_y})\n"
                 f"Flux drop: {flux_drop_percent:.1f}%")

    # Place the text in the chosen corner
    ax.text(text_x, text_y, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment=text_va, horizontalalignment=text_ha,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # Inset plot for the light curve
    # Place it using the chosen inset_loc
    inset_ax = inset_axes(ax, width="40%", height="30%", loc=inset_loc, borderpad=1)
    inset_ax.plot(range(1, M+1), rel_flux, marker='o', color='red', linestyle='-')
    inset_ax.set_title("Transit Light Curve", fontsize=8, color='white')
    inset_ax.set_xlabel("Image #", fontsize=8, color='white')
    inset_ax.set_ylabel("Rel. Flux", fontsize=8, color='white')
    inset_ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig("transiting_exoplanet_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
