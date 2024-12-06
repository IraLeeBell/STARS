import sys
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import math

def generate_exoplanet_stack(sx, sy, NN, M, sigma=2.0, RN=10.0, sky_lev=1000.0):
    edge = 10
    im0 = np.zeros((sx, sy), dtype='float32')

    m_min, m_max = 9.0, 20.27
    cn = np.random.rand(NN)
    m = 2 * np.log(np.exp(0.5*m_min) + cn*(np.exp(0.5*m_max) - np.exp(0.5*m_min)))
    x = edge + (np.random.rand(NN) * (sx - 2*edge)).astype('int16')
    y = edge + (np.random.rand(NN) * (sy - 2*edge)).astype('int16')
    star_coords = list(zip(x, y))

    snr_min = 10.0
    noise = np.sqrt(RN**2 + sky_lev) * np.sqrt(4 * math.pi) * sigma
    cts = snr_min * noise * 10**(-0.4*(m - m_max))
    snr = cts / np.sqrt(cts + noise**2)

    bright_stars = np.where(snr > 100)[0]
    if len(bright_stars) == 0:
        transit_star_idx = np.argmax(cts)
    else:
        transit_star_idx = np.random.choice(bright_stars)

    # Pick a different reference star
    possible_refs = bright_stars[bright_stars != transit_star_idx]
    if len(possible_refs) == 0:
        ref_candidates = np.arange(NN)
        ref_candidates = ref_candidates[ref_candidates != transit_star_idx]
        ref_star_idx = np.random.choice(ref_candidates)
    else:
        ref_star_idx = np.random.choice(possible_refs)

    im_s = np.zeros((sx, sy), dtype='float32')
    for i in range(NN):
        im_s[x[i], y[i]] += cts[i]
    im_s = gaussian_filter(im_s, sigma)

    # Define a deeper transit: from full flux (1.0) down to 0.7 (30% dip)
    dip_level = 0.7
    ingress_start = M // 3
    ingress_end = M // 2
    egress_start = M // 2
    egress_end = (2 * M) // 3

    flux_multiplier = np.ones(M)
    for i in range(M):
        if ingress_start <= i < ingress_end:
            frac = (i - ingress_start) / (ingress_end - ingress_start)
            flux_multiplier[i] = 1.0 - (1.0 - dip_level)*frac
        elif ingress_end <= i < egress_start:
            flux_multiplier[i] = dip_level
        elif egress_start <= i < egress_end:
            frac = (i - egress_start) / (egress_end - egress_start)
            flux_multiplier[i] = dip_level + (1.0 - dip_level)*frac

    images = []
    transit_cts = cts[transit_star_idx]
    transit_star_x, transit_star_y = star_coords[transit_star_idx]
    transit_flux = []
    # for i in range(M):
    #     frame = im_s.copy()
    #     delta_flux = (flux_multiplier[i] - 1.0)*transit_cts
    #     frame[transit_star_x, transit_star_y] += delta_flux
    #     frame_with_noise = frame + np.sqrt(RN**2 + frame + sky_lev)*np.random.randn(sx, sy)
    #     images.append(frame_with_noise.astype('float32'))
    #     transit_flux.append(transit_cts * flux_multiplier[i])

    for i in range(M):
        frame = im_s.copy()
        delta_flux = (flux_multiplier[i] - 1.0)*transit_cts
        frame[transit_star_x, transit_star_y] += delta_flux

        # Ensure we do not take sqrt of a negative number:
        noise_term = np.sqrt(np.maximum(RN**2 + frame + sky_lev, 0.0)) * np.random.randn(sx, sy)
        frame_with_noise = frame + noise_term

        images.append(frame_with_noise.astype('float32'))
        transit_flux.append(transit_cts * flux_multiplier[i])


    return images, star_coords, transit_star_idx, ref_star_idx, np.array(transit_flux)

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        sx = np.random.randint(512, 2048)
        sy = np.random.randint(512, 2048)
        NN = np.random.randint(100, 1000)
        M = 10
    elif len(args) == 3:
        sx = int(args[0])
        sy = int(args[1])
        NN = int(args[2])
        M = 10
    elif len(args) == 4:
        sx = int(args[0])
        sy = int(args[1])
        NN = int(args[2])
        M = int(args[3])
    else:
        print("Usage: python generate_exoplanet_images.py [sx sy stars [M]]")
        sys.exit(1)

    images, star_coords, transit_star_idx, ref_star_idx, transit_flux = generate_exoplanet_stack(sx, sy, NN, M)

    for i, img in enumerate(images, start=1):
        filename = f"exoplanet_image_{str(i).zfill(4)}.fits"
        fits.writeto(filename, img, overwrite=True)
        print(f"Generated {filename}")

    # Save star info to a file
    np.savez("exoplanet_star_info.npz", 
             star_coords=star_coords, 
             transit_star_idx=transit_star_idx, 
             ref_star_idx=ref_star_idx,
             sx=sx, sy=sy, M=M, NN=NN)
    print("Saved star info to exoplanet_star_info.npz")

    print("Done!")
    print(f"Transit star index: {transit_star_idx}, position: {star_coords[transit_star_idx]}")
    print(f"Reference star index: {ref_star_idx}, position: {star_coords[ref_star_idx]}")

if __name__ == "__main__":
    main()
