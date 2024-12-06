from numpy import exp,log,zeros,vstack,sqrt,pi,zeros,load
from numpy.random import rand,randn
from scipy.ndimage import gaussian_filter

def imsim(sigma=2.,RN=10.,sky_lev=1000.,add_stars=True,calibrated=True,verbose=False):
    """
        generate astronomical images, potentially with stars (add_stars=True)

        calibrated=False means image is uncorrected for flat and dark, to estimate these:
          bias images: set sky_lev=0.,add_stars=False
          flat images: set sky_lev large, add_stars=False

        sigma: rms width of stars in pixels
        RN: rms read noise
        sky_lev: sky background counts level
    """
    sx,sy,edge = 1024,1024,10

    # start with read-noise
    im = RN*randn(sx,sy)

    if (not calibrated):
        if (verbose): print (" Applying bias-field image")
        im += load('/srv/data/ses_350_data/bias.npy')

    im_s=0.
    if (add_stars):
        # a simple logN ~ -0.5*logS

        snr_min=10.
        m_min,m_max = 9.,20.27
        N = int( (0.25/100.)*( 2*( exp(0.5*m_max)-exp(0.5*m_min) ) ) )

        if (verbose):
            print ("""Simulating {:d}x{:d} image with {:d} stars (mag={:.2f}-{:.2f})""".format(sx,sy,N,m_min,m_max))

        cn = rand(N)
        m = 2*log( exp(0.5*m_min) + cn*( exp(0.5*m_max)-exp(0.5*m_min) ) )

        x = edge + (rand(N)*(sx-2*edge)).astype('int16')
        y = edge + (rand(N)*(sy-2*edge)).astype('int16')
        cts = snr_min*sqrt(RN**2+sky_lev)*10**(-0.4*(m-m_max))*sqrt(4*pi)*sigma

        im_s = zeros((sx,sy),dtype='float32')
        for i in range(N):
            im_s[x[i],y[i]] += cts[i]

        im_s = gaussian_filter(im_s,sigma)

    flt=1.
    if (not calibrated):
        if (verbose): print (" Applying flat-field image")
        flt = load('/srv/data/ses_350_data/flat.npy')

    im += ( im_s + sky_lev + sqrt(im_s+sky_lev)*randn(sx,sy) )*flt

    if (add_stars): return im, vstack((x,y,cts))
    else: return im
