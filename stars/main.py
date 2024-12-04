import numpy as np
from scipy.fftpack import fft

from find_stars import find_stars
from measure_stars import measure_stars
from simulate_image_stack import imsim_multi

def find_periodic_source(data):
    im0 = np.mean(data, axis=0)
    
    # Correct order: sx (columns), sy (rows)
    sx, sy = find_stars(im0)
    
    N, M = len(data), len(sx)
    flux = np.zeros((N, M), dtype='float32')
    
    for i in range(N):
        flux[i] = measure_stars(data[i], sx, sy)
    
    ff = flux.mean(axis=0)
    vff = flux.var(axis=0)
    
    # Compute PSD
    psd = np.abs(fft(flux - ff, axis=0)[:N//2])**2 / (N * vff)
    psd[0, :] = 0  # Exclude zero frequency component
    
    # Find maximum PSD and indices
    psdm = np.max(psd)
    i, j = np.unravel_index(np.argmax(psd), psd.shape)
    
    # Compute period and amplitude
    period = N / i
    amplitude = np.sqrt(psdm * vff[j] / N) * 2 / ff[j]
    
    return j, period, amplitude

if __name__ == "__main__":
    data = imsim_multi()
    j, period, ampl = find_periodic_source(data)
    print("Source {:d} Period and Amplitude: {:.1f}, {:.1f}".format(j, period, ampl))
