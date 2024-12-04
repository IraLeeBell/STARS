import numpy as np
from scipy.ndimage import gaussian_filter

def imsim():
    sx,sy = 1024,1024

    # start with read-noise
    im = 10.*np.random.randn(sx,sy)

    m_min,m_max = 9.,20.27
    N = int( (0.25/100.)*( 2*( np.exp(0.5*m_max)-np.exp(0.5*m_min) ) ) )

    cn = np.random.rand(N)
    m = 2*np.log( np.exp(0.5*m_min) + cn*( np.exp(0.5*m_max)-np.exp(0.5*m_min) ) )

    x = 10+(np.random.rand(N)*(sx-20)).astype('int16')
    y = 10+(np.random.rand(N)*(sy-20)).astype('int16')
    cts = 10*np.sqrt(1.1e3)*10**(-0.4*(m-m_max))*np.sqrt(4*np.pi)*2

    im_s = np.zeros((sx,sy),dtype='float32')
    for i in range(N):
        im_s[x[i],y[i]] += cts[i]

    return gaussian_filter(im_s,2.) + np.sqrt(im_s+1.e3)*np.random.randn(sx,sy), x , y