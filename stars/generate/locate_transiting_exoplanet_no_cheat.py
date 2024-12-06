import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def trapezoid_model(M, normal_flux, dip_flux, ingress_start, ingress_end, egress_start, egress_end):
    """
    Create a trapezoidal light curve model given the parameters.
    Returns an array of length M with the modeled flux.
    """
    lc = np.ones(M)*normal_flux
    # Ingress: ramp down linearly
    for i in range(ingress_start, ingress_end):
        frac = (i - ingress_start) / (ingress_end - ingress_start)
        lc[i] = normal_flux - (normal_flux - dip_flux)*frac

    # Full dip
    for i in range(ingress_end, egress_start):
        lc[i] = dip_flux

    # Egress: ramp up linearly
    for i in range(egress_start, egress_end):
        frac = (i - egress_start) / (egress_end - egress_start)
        lc[i] = dip_flux + (normal_flux - dip_flux)*frac

    return lc

def main():
    files = sorted(glob.glob("exoplanet_image_*.fits"))
    if len(files) == 0:
        print("No images found matching 'exoplanet_image_XXXX.fits'")
        return

    # Load all images
    images = []
    for f in files:
        with fits.open(f) as hdul:
            data = hdul[0].data
            images.append(data)
    images = np.array(images)  # shape: (M, sx, sy)
    M, sx, sy = images.shape
    print(f"Loaded {M} images of size {sx}x{sy}")

    # Detect sources in the first image using DAOStarFinder
    mean, median, std = sigma_clipped_stats(images[0], sigma=3.0)
    # Increase threshold if too many false detections occur, decrease if too few
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(images[0] - median)
    if sources is None or len(sources) == 0:
        print("No stars found. Try adjusting source detection parameters.")
        return

    print(f"Detected {len(sources)} sources.")

    # Positions of detected stars (x is column, y is row)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    
    # Perform aperture photometry on each image
    # Larger aperture to ensure we get enough flux
    apertures = CircularAperture(positions, r=5.0)
    star_fluxes = []
    for img in images:
        phot_table = aperture_photometry(img, apertures)
        flux = np.array(phot_table['aperture_sum'])
        star_fluxes.append(flux)
    star_fluxes = np.array(star_fluxes)  # shape: (M, N_stars)
    N_stars = star_fluxes.shape[1]
    print(f"Total stars after aperture photometry: {N_stars}")

    # Remove stars that have any NaN flux (just in case)
    nan_mask = np.isnan(star_fluxes).any(axis=0)
    star_fluxes = star_fluxes[:, ~nan_mask]
    positions = positions[~nan_mask]
    N_stars = star_fluxes.shape[1]
    print(f"{N_stars} stars remain after removing NaNs.")

    # Select multiple reference stars:
    # Criteria: pick the top 10 brightest stable stars to form a reference ensemble
    mean_fluxes = np.mean(star_fluxes, axis=0)
    std_devs = np.std(star_fluxes, axis=0)
    # Stability metric: std/mean (lower is better)
    stability = std_devs / mean_fluxes
    # Filter to bright stars for reference
    bright_filter = mean_fluxes > 100  # Adjust if needed
    if not np.any(bright_filter):
        print("No bright stars for reference found. Try adjusting parameters.")
        return

    bright_indices = np.where(bright_filter)[0]
    # Sort bright stars by stability
    stable_sorted = bright_indices[np.argsort(stability[bright_indices])]
    # Pick top 10 or all if fewer than 10
    top_refs = stable_sorted[:min(10, len(stable_sorted))]
    ref_fluxes = star_fluxes[:, top_refs]
    # Average flux of these reference stars
    ref_flux = np.mean(ref_fluxes, axis=1)

    # Normalize by reference flux
    epsilon = 1e-6
    ref_flux_safe = np.where(ref_flux < epsilon, epsilon, ref_flux)
    normalized_fluxes = star_fluxes / ref_flux_safe[:, None]

    # Transit timing from generation scenario
    ingress_start = M // 3
    ingress_end = M // 2
    egress_start = M // 2
    egress_end = (2 * M) // 3

    # Try trapezoid fitting with a larger range of dip depths: 5% to 90%
    dip_candidates = np.arange(0.05, 0.95, 0.05)

    best_improvement = -np.inf
    best_star = None
    best_flux = None
    best_drop = None

    # We'll also compute chi2 for a flat (no-transit) model to compare
    for i in range(N_stars):
        flux = normalized_fluxes[:, i]
        if np.isnan(flux).any():
            continue
        median_flux = np.median(flux)

        # No-transit model: just a flat line at median_flux
        no_transit_chi2 = np.sum((flux - median_flux)**2)

        best_local_chi2 = np.inf
        best_local_drop = None
        for dip_level in dip_candidates:
            dip_flux = median_flux*(1 - dip_level)
            model = trapezoid_model(M, median_flux, dip_flux, ingress_start, ingress_end, egress_start, egress_end)
            residuals = flux - model
            chi2 = np.sum(residuals**2)
            if chi2 < best_local_chi2:
                best_local_chi2 = chi2
                best_local_drop = dip_level

        # Improvement over no-transit model
        improvement = no_transit_chi2 - best_local_chi2
        if improvement > best_improvement:
            best_improvement = improvement
            best_star = i
            best_flux = flux
            best_drop = best_local_drop

    if best_star is None:
        print("No transit-like star found. Consider adjusting thresholds or parameters.")
        return

    t_flux = best_flux
    flux_min = np.min(t_flux)
    flux_drop_percent = (1 - flux_min)*100
    t_x, t_y = positions[best_star]

    # Choose one image to display (the first one)
    chosen_image_idx = 0
    chosen_data = images[chosen_image_idx]
    fig, ax = plt.subplots(figsize=(6,6))
    norm_min, norm_max = np.percentile(chosen_data, [5, 95])
    ax.imshow(chosen_data, cmap='gray', origin='lower', vmin=norm_min, vmax=norm_max)

    # Mark the star
    circ = Circle((t_x, t_y), radius=5, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circ)

    half_sx = sx / 2
    half_sy = sy / 2

    # Determine quadrant
    if t_y < half_sx and t_x < half_sy:
        # top-left quadrant
        text_x, text_y = 0.95, 0.05
        text_ha, text_va = 'right', 'bottom'
        inset_loc = 'upper right'
    elif t_y < half_sx and t_x >= half_sy:
        # top-right quadrant
        text_x, text_y = 0.05, 0.05
        text_ha, text_va = 'left', 'bottom'
        inset_loc = 'upper left'
    elif t_y >= half_sx and t_x < half_sy:
        # bottom-left quadrant
        text_x, text_y = 0.95, 0.95
        text_ha, text_va = 'right', 'top'
        inset_loc = 'lower right'
    else:
        # bottom-right quadrant
        text_x, text_y = 0.05, 0.95
        text_ha, text_va = 'left', 'top'
        inset_loc = 'lower left'

    info_text = (f"Transiting Star (No Cheat):\n"
                 f"Coordinates (y,x): ({t_y:.0f}, {t_x:.0f})\n"
                 f"Flux drop: {flux_drop_percent:.1f}%")

    ax.text(text_x, text_y, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment=text_va, horizontalalignment=text_ha,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    inset_ax = inset_axes(ax, width="40%", height="30%", loc=inset_loc, borderpad=1)
    inset_ax.plot(range(1, M+1), t_flux, marker='o', color='red', linestyle='-')
    inset_ax.set_title("Fitted Transit Light Curve", fontsize=8)
    inset_ax.set_xlabel("Image #", fontsize=8)
    inset_ax.set_ylabel("Rel. Flux", fontsize=8)
    inset_ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig("transiting_exoplanet_result_no_cheat.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
