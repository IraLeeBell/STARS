from numpy import exp,log,zeros,sqrt,pi,zeros,where,sin
from numpy.random import rand,randn,randint
from scipy.ndimage import gaussian_filter

def imsim_multi(sigma=2.,RN=10.,sky_lev=1000.,N=30):
    """
        generate set of N astronomical images

        randomly selects and makes periodic one of the stars

        sigma: rms width of stars in pixels
        RN: rms read noise
        sky_lev: sky background counts level
    """
    sx,sy,edge = 512,512,10

    # start with an empty image
    im0 = zeros((sx,sy),dtype='float32')

    ##########################################
    # add stars from a simple logN ~ -0.5*logS
    ##########################################

    m_min,m_max = 9.,20.27
    NN = int( 0.25*(0.25/100.)*( 2*( exp(0.5*m_max)-exp(0.5*m_min) ) ) )

    cn = rand(NN)
    m = 2*log( exp(0.5*m_min) + cn*( exp(0.5*m_max)-exp(0.5*m_min) ) )

    x = edge + (rand(NN)*(sx-2*edge)).astype('int16')
    y = edge + (rand(NN)*(sy-2*edge)).astype('int16')

    snr_min=10.
    noise = sqrt(RN**2+sky_lev)*sqrt(4*pi)*sigma
    cts = snr_min*noise*10**(-0.4*(m-m_max))
    snr = cts/sqrt(cts+noise**2)

    # select a star to make periodic
    ii = where((snr>100.)*(snr<1000.))[0]
    i0 = ii[int(rand()*len(ii))]
    f = randint(N/5-2,N/5+2)*1./N

    im_s0 = zeros((sx,sy),dtype='float32')
    im_s0[x[i0],y[i0]] = cts[i0]

    im_s = zeros((sx,sy),dtype='float32')
    for i in range(NN):
        if (i==i0): continue
        im_s[x[i],y[i]] += cts[i]

    im_s = gaussian_filter(im_s,sigma)
    im_s0 = gaussian_filter(im_s0,sigma)

    ##########################################

    # add noise and create a list of N images
    im=[]
    x0 = rand()*N
    for i in range(N):
        im_s1 = im_s0*(1 + 0.1*sin(2*pi*f*(i-x0)))
        im.append( im0 + ( im_s + im_s1 + sqrt(RN**2+im_s+sky_lev)*randn(sx,sy) ) )

    return im
