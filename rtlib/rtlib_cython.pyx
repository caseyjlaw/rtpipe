import numpy as n
cimport numpy as n
cimport cython
#import logging
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

# can choose between numpy and pyfftw
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft   # numpy -> NUMPY for NRAO install
    ffttype = 'pyfftw'
except:
    from numpy import fft
    ffttype = 'numpy'
    
CTYPE = n.long
ctypedef n.long_t CTYPE_t
DTYPE = n.complex64
ctypedef n.complex64_t DTYPE_t

cpdef beamonefullxy(n.ndarray[n.float32_t, ndim=2, mode='c'] u, n.ndarray[n.float32_t, ndim=2, mode='c'] v, n.ndarray[DTYPE_t, ndim=3, mode='c'] data, unsigned int npixx, unsigned int npixy, unsigned int res):
    # Same as imgonefullxy, but returns dirty beam
    # Ignores uv points off the grid
    # flips xy gridding! im on visibility flux scale!
    # on flux scale (counts nonzero data)

    # initial definitions
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef int cellu
    cdef int cellv
    cdef unsigned int nonzeros = 0
    cdef n.ndarray[DTYPE_t, ndim=2] grid = n.zeros( (npixx,npixy), dtype='complex64')
    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    cdef arr = pyfftw.n_byte_align_empty((npixx,npixy), 16, dtype='complex64')
    ifft = pyfftw.builders.ifft2(arr, overwrite_input=True, auto_align_input=True, auto_contiguous=True)

    ok = n.logical_and(n.abs(uu) < npixx/2, n.abs(vv) < npixy/2)
    uu = n.mod(uu, npixx)
    vv = n.mod(vv, npixy)

    # add uv data to grid
    # or use np.add.at(x, i, y)?
    for i in xrange(len0):
        for j in xrange(len1):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for p in xrange(len2):
                    if data[i,j,p] != 0j:
                        grid[cellu, cellv] = 1 + grid[cellu, cellv] 
                        nonzeros = nonzeros + 1

    # make images and filter based on threshold
    arr[:] = grid[:]
    im = ifft(arr).real*int(npixx*npixy)/float(nonzeros)
    im = recenter(im, (npixx/2,npixy/2))

    print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(npixx*npixy)/float(nonzeros))
    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./(npixx*res)), 3600*n.degrees(2./(npixy*res)), 3600*n.degrees(1./res))
    return im

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgonefullxy(n.ndarray[n.float32_t, ndim=2, mode='c'] u, n.ndarray[n.float32_t, ndim=2, mode='c'] v, n.ndarray[DTYPE_t, ndim=3, mode='c'] data, unsigned int npixx, unsigned int npixy, unsigned int uvres, verbose=1):
    # Same as imgallfullxy, but one flux scaled image
    # Defines uvgrid filter before loop
    # flips xy gridding!

    # initial definitions
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef unsigned int nonzeros = 0
    cdef n.ndarray[DTYPE_t, ndim=2] grid = n.zeros((npixx,npixy), dtype='complex64')
    cdef arr = pyfftw.n_byte_align_empty((npixx,npixy), 16, dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/uvres).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/uvres).astype(n.int)

    ifft = pyfftw.builders.ifft2(arr, overwrite_input=True, auto_align_input=True, auto_contiguous=True)
    
    ok = n.logical_and(n.abs(uu) < npixx/2, n.abs(vv) < npixy/2)
    uu = n.mod(uu, npixx)
    vv = n.mod(vv, npixy)

    # add uv data to grid
    # or use np.add.at(x, i, y)?
    for i in xrange(len0):
        for j in xrange(len1):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for p in xrange(len2):
                    grid[cellu, cellv] = data[i,j,p] + grid[cellu, cellv]
                    if data[i,j,p] != 0j:
                        nonzeros = nonzeros + 1

    # make images and filter based on threshold
    arr[:] = grid[:]
    im = ifft(arr).real*int(npixx*npixy)
    im = recenter(im, (npixx/2,npixy/2))
    
    if nonzeros > 0:
        im = im/float(nonzeros)
        if verbose:
            print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(npixx*npixy)/float(nonzeros))
    else:
        if verbose:
            print 'Gridded %.3f of data. All zeros.' % (float(ok.sum())/ok.size)
    if verbose:
        print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./(npixx*uvres)), 3600*n.degrees(2./(npixy*uvres)), 3600*n.degrees(1./uvres))
    return im

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfullfilterxy(n.ndarray[n.float32_t, ndim=2, mode='c'] u, n.ndarray[n.float32_t, ndim=2, mode='c'] v, n.ndarray[DTYPE_t, ndim=4, mode='c'] data, unsigned int npixx, unsigned int npixy, unsigned int res, float thresh):
    # Same as imgallfull, but returns both pos and neg candidates
    # Defines uvgrid filter before loop
    # flips xy gridding!

    # initial definitions
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,npixx,npixy), dtype='complex64')
    cdef arr = pyfftw.n_byte_align_empty((npixx,npixy), 32, dtype='complex64')
    cdef float snr

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ifft = pyfftw.builders.ifft2(arr, overwrite_input=True, auto_align_input=True, auto_contiguous=True, planner_effort='FFTW_PATIENT')
    
    ok = n.logical_and(n.abs(uu) < npixx/2, n.abs(vv) < npixy/2)
    uu = n.mod(uu, npixx)
    vv = n.mod(vv, npixy)

    # add uv data to grid
    # or use np.add.at(x, i, y)?
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    # make images and filter based on threshold
    candints = []; candims = []; candsnrs = []
    for t in xrange(len0):
        arr[:] = grid[t]
        im = ifft(arr).real

        # find most extreme pixel
        snrmax = im.max()/im.std()
        snrmin = im.min()/im.std()
        if snrmax >= abs(snrmin):
            snr = snrmax
        else:
            snr = snrmin
        if ( (abs(snr) > thresh) & n.any(data[t,:,len2/3:,:])):
            candints.append(t)
            candsnrs.append(snr)
            candims.append(recenter(im, (npixx/2,npixy/2)))

#    print 'Detected %d candidates with at least third the band.' % len(candints)
#    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./(npixx*res)), 3600*n.degrees(2./(npixy*res)), 3600*n.degrees(1./res))
    return candims,candsnrs,candints

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfullfilterxyflux(n.ndarray[n.float32_t, ndim=2, mode='c'] u, n.ndarray[n.float32_t, ndim=2, mode='c'] v, n.ndarray[DTYPE_t, ndim=4, mode='c'] data, unsigned int npixx, unsigned int npixy, unsigned int res, float thresh):
    # Same as imgallfull, but returns only candidates and rolls images
    # Defines uvgrid filter before loop
    # flips xy gridding!
    # counts nonzero data and properly normalizes fft to be on flux scale

    # initial definitions
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef unsigned int nonzeros = 0
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,npixx,npixy), dtype='complex64')
    cdef arr = pyfftw.n_byte_align_empty((npixx,npixy), 16, dtype='complex64')
    cdef float snr

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ifft = pyfftw.builders.ifft2(arr, overwrite_input=True, auto_align_input=True, auto_contiguous=True, planner_effort='FFTW_PATIENT')
    
    ok = n.logical_and(n.abs(uu) < npixx/2, n.abs(vv) < npixy/2)
    uu = n.mod(uu, npixx)
    vv = n.mod(vv, npixy)

    # add uv data to grid
    # or use np.add.at(x, i, y)?
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    # make images and filter based on threshold
    candints = []; candims = []; candsnrs = []
    for t in xrange(len0):
        arr[:] = grid[t]
        im = ifft(arr).real*int(npixx*npixy)

        # find most extreme pixel
        snrmax = im.max()/im.std()
        snrmin = im.min()/im.std()
        if snrmax >= abs(snrmin):
            snr = snrmax
        else:
            snr = snrmin
        if ( (abs(snr) > thresh) & n.any(data[t,:,len2/3:,:])):

            # calculate number of nonzero vis to normalize fft
            nonzeros = 0
            for i in xrange(len1):
                for j in xrange(len2):
                    if ok[i,j]:
                        for p in xrange(len3):
                            if data[t,i,j,p] != 0j:
                                nonzeros = nonzeros + 1

            candints.append(t)
            candsnrs.append(snr)
            candims.append(recenter(im/float(nonzeros), (npixx/2,npixy/2)))

    return candims,candsnrs,candints

cpdef imgonefullw(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=3] data, unsigned int npix, unsigned int uvres, blsets, kers, verbose=1):
    # Same as imgallfullxy, but includes w term

    # initial definitions
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef unsigned int kernum
    cdef n.ndarray[n.int_t, ndim=1] bls
    cdef int keru
    cdef int kerv
    cdef n.ndarray[DTYPE_t, ndim=2] grid = n.zeros((npix,npix), dtype='complex64')
#    cdef n.ndarray[DTYPE_t, ndim=2] gridacc = n.zeros((npix,npix), dtype='complex64')
    cdef arr = pyfftw.n_byte_align_empty((npix,npix), 16, dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/uvres).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/uvres).astype(n.int)
    cdef n.ndarray[DTYPE_t, ndim=2] ker
    cdef int ksize

    ifft = pyfftw.builders.ifft2(arr)
#    fft = pyfftw.builders.fft2(arr)
    
    uu = n.mod(uu, npix)
    vv = n.mod(vv, npix)

    # add uv data to grid
    for kernum in xrange(len(kers)):
        bls = blsets[kernum]
        ker = kers[kernum]
        ksize = len(ker)
        ok = n.logical_and(uu < (npix-ksize), vv < (npix-ksize))
        for i in bls:
            for j in xrange(len1):
                if ok[i,j]:
                    cellu = uu[i,j]
                    cellv = vv[i,j]
                    for p in xrange(len2):
# option 0) no w-term correction
#                        grid[cellu, cellv] = data[i,j,p] + grid[cellu, cellv]  # no conv gridding
#                        gridacc[cellu, cellv] = data[i,j,p] + gridacc[cellu, cellv]  # no conv gridding
# option 1) conv gridding
                         for keru in xrange(ksize):
                             for kerv in xrange(ksize):
                                 grid[cellu+keru-ksize/2, cellv+kerv-ksize/2] = ker[keru,kerv]*data[i,j,p] + grid[cellu+keru-ksize/2, cellv+kerv-ksize/2]
# option 2) fourier kernel (not working)
#        print 'kernum:', kernum
#        arr[:] = gridacc[:]
#        imacc = fft()
#        arr[:] = (imacc*ker)[:]   
#        grid[:] += ifft()[:]

    # make images and filter based on threshold
    arr[:] = grid[:]
    im = ifft().real
    im = recenter(im, (npix/2,npix/2))

    if verbose:
        print 'Pixel size %.1f\". Field size %.1f\"' % (3600*n.degrees(2./(npix*uvres)), 3600*n.degrees(1./uvres))
    return im

cpdef genuvkernels(n.ndarray[n.float32_t, ndim=1] w, wres, unsigned int npix, unsigned int uvres, float thresh=0.99, unsigned int oversample=2, unsigned int ksize=0):
    cdef n.ndarray[DTYPE_t, ndim=2] uvker
    npix = npix*oversample
    uvres = uvres/oversample
    cdef lmker = pyfftw.n_byte_align_empty( (npix, npix), 16, dtype='complex64')

    dofft = pyfftw.builders.fft2(lmker)

    # set up w planes
    sqrt_w = n.sqrt(n.abs(w)) * n.sign(w)
    numw = n.ceil(1.1*(sqrt_w.max()-sqrt_w.min())/wres).astype(int)  # number of w bins (round up)
    wgrid = n.linspace(sqrt_w.min()*1.05, sqrt_w.max()*1.05, numw)

    # Grab a chunk of uvw's that grid w to same point.
    uvkers = []; blsets = []
    for i in range(len(wgrid)-1):
        # get baselines in this wgrid bin
        blw = n.where( (sqrt_w > wgrid[i]) & (sqrt_w <= wgrid[i+1]) )
        blsets.append(blw[0])
        avg_w = n.average(w[blw])
        print 'w %.1f to %.1f: Added %d/%d baselines' % (n.sign(wgrid[i])*wgrid[i]**2, n.sign(wgrid[i+1])*wgrid[i+1]**2, len(blw[0]), len(w))

        if len(blw[0]):
            # get image extent
            lmker[:] = get_lmkernel(npix, uvres, avg_w).astype(DTYPE)

            # uv kernel from inv fft of lm kernel
            im = dofft()
            uvker = recenter(im, (npix/2,npix/2))

            if ksize == 0:
                # keep uvker above a fraction (thresh) of peak amp
                largey, largex = n.where(n.abs(uvker) > thresh*n.abs(uvker).max())
                ksize = max(largey.max()-largey.min(), largex.max()-largex.min())                # take range of high values to define kernel size
                uvker = uvker[npix/2-ksize/2:npix/2+ksize/2+1, npix/2-ksize/2:npix/2+ksize/2+1]
                uvkers.append((uvker/uvker.sum()).astype(DTYPE))
            else:
                uvker = uvker[npix/2-ksize/2:npix/2+ksize/2+1, npix/2-ksize/2:npix/2+ksize/2+1]
                uvkers.append((uvker/uvker.sum()).astype(DTYPE))
        else:
            uvkers.append([])

    return blsets, uvkers

cpdef genlmkernels(n.ndarray[n.float32_t, ndim=1] w, wres, unsigned int npix, unsigned int uvres):

    # set up w planes
    sqrt_w = n.sqrt(n.abs(w)) * n.sign(w)
    numw = n.ceil(1.1*(sqrt_w.max()-sqrt_w.min())/wres).astype(int)  # number of w bins (round up)
    wgrid = n.linspace(sqrt_w.min()*1.05, sqrt_w.max()*1.05, numw)

    # Grab a chunk of uvw's that grid w to same point.
    lmkers = []; blsets = []
    for i in range(len(wgrid)-1):
        # get baselines in this wgrid bin
        blw = n.where( (sqrt_w > wgrid[i]) & (sqrt_w <= wgrid[i+1]) )
        blsets.append(blw[0])
        avg_w = n.average(w[blw])
        print 'w %.1f to %.1f: Added %d/%d baselines' % (n.sign(wgrid[i])*wgrid[i]**2, n.sign(wgrid[i+1])*wgrid[i+1]**2, len(blw[0]), len(w))

        if len(blw[0]):
            lmkers.append(get_lmkernel(npix, uvres, avg_w).astype(DTYPE))
        else:
            lmkers.append([])

    return blsets, lmkers

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_lmkernel(npix, res, avg_w):
    l, m = get_lm(npix, res)
    sqrtn = n.sqrt(1 - l**2 - m**2).astype(n.float32)
    G = n.exp(-2*n.pi*1j*avg_w*(sqrtn - 1))
    G = G.filled(0)
    # Unscramble difference between fft(fft(G)) and G
    G[1:] = n.flipud(G[1:]).copy()
    G[:,1:] = n.fliplr(G[:,1:]).copy()
    return G / G.size

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_lm(npix, res, center=(0,0)):
    m,l = n.indices((npix,npix))
    l,m = n.where(l > npix/2, npix-l, -l), n.where(m > npix/2, m-npix, m)
    l,m = l.astype(n.float32)/npix/res, m.astype(n.float32)/npix/res
    mask = n.where(l**2 + m**2 >= 1, 1, 0)
    l,m = n.ma.array(l, mask=mask), n.ma.array(m, mask=mask)
    return recenter(l, center), recenter(m, center)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef recenter(a, c):
    s = a.shape
    c = (c[0] % s[0], c[1] % s[1])
    if n.ma.isMA(a):
        a1 = n.ma.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = n.ma.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    else:
        a1 = n.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = n.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    return a2

def sigma_clip(n.ndarray[n.float32_t, ndim=1] arr, float sigma=3):
    """ Function takes 1d array of values and returns the sigma-clipped min and max scaled by value "sigma".
    """

    assert arr.dtype == n.float32

    cdef n.ndarray[n.int_t, ndim=1] cliparr = n.arange(len(arr))
    cdef float mean = 0
    cdef float std = 0

    arr = n.append(arr,[n.float32(1)])    # append superfluous item to trigger loop
    while len(cliparr) != len(arr):
        arr = arr[cliparr]
        mean = arr.mean()
        std = arr.std()
        cliparr = n.where((arr < mean + sigma*std) & (arr > mean - sigma*std) & (arr != 0) )[0]
#        print 'Clipping %d from array of length %d' % (len(arr) - len(cliparr), len(arr))
    return mean - sigma*std, mean + sigma*std

cpdef make_triples(d):
    """ Calculates and returns data indexes (i,j,k) for all closed triples.
    """

    ants = d['ants']
    nants = d['nants']
    blarr = calc_blarr(d)

    cdef int t
    cdef int ant1
    cdef int ant2
    cdef int ant3
    cdef n.ndarray[n.int_t, ndim=2] triples = n.zeros((nants*(nants-1)*(nants-2)/6, 3), dtype='int')

    # first make triples indexes in antenna numbering
    anttrips = []
    for i in ants:
        for j in ants[list(ants).index(i)+1:]:
            for k in ants[list(ants).index(j)+1:]:
                anttrips.append([i,j,k])
                
    # next return data indexes for triples
    for t in xrange(len(anttrips)):
        ant1 = anttrips[t][0]
        ant2 = anttrips[t][1]
        ant3 = anttrips[t][2]
        try:
            bl1 = n.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant2) )[0][0]
            bl2 = n.where( (blarr[:,0] == ant2) & (blarr[:,1] == ant3) )[0][0]
            bl3 = n.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant3) )[0][0]
            triples[t,0] = bl1
            triples[t,1] = bl2
            triples[t,2] = bl3
        except IndexError:
            continue

    return triples

@cython.profile(False)
cdef n.ndarray[DTYPE_t, ndim=2] fringe_rotation(float dl, float dm, n.ndarray[n.float32_t, ndim=1] u, n.ndarray[n.float32_t, ndim=1] v, n.ndarray[n.float32_t, ndim=1] freq): 
    return n.exp(-2j*3.1415*(dl*n.outer(u,freq) + dm*n.outer(v,freq)))

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef phaseshift(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, d, float l1, float m1, n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, verbose=0):
    """ Shift phase center to (l1, m1).
    Assumes single uv over all times in data. Reasonable for up to a second or so of data.
    """

    cdef n.ndarray[complex, ndim=2] frot
    cdef n.ndarray[float, ndim=1] freq = d['freq_orig'][d['chans']]
    cdef n.ndarray[float, ndim=1] freq_orig = d['freq_orig']
    cdef float dl = l1 - d['l0']
    cdef float dm = m1 - d['m0']
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l

    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]

    if (dl != 0.) or (dm != 0.):
        frot = fringe_rotation(dl, dm, u[len0/2], v[len0/2], freq/freq_orig[0])
        for j in xrange(len1):
            for i in xrange(len0):
                for l in xrange(len3):    # iterate over pols
                    for k in xrange(len2):
                        data[i,j,k,l] = data[i,j,k,l] * frot[j,k]
    else:
        if verbose:
            print 'No phase rotation needed'

    d['l0'] = l1
    d['m0'] = m1

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef phaseshift_threaded(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, d, float l1, float m1, n.ndarray[n.float32_t, ndim=1, mode='c'] u, n.ndarray[n.float32_t, ndim=1, mode='c'] v, verbose=0):
    """ Shift phase center to (l1, m1).
    Assumes single uv over all times in data. Reasonable for up to a second or so of data.
    """

    cdef n.ndarray[DTYPE_t, ndim=2] frot
    cdef n.ndarray[float, ndim=1] freq = d['freq_orig'][d['chans']]
    cdef n.ndarray[float, ndim=1] freq_orig = d['freq_orig']
    cdef float dl = l1 - d['l0']
    cdef float dm = m1 - d['m0']
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l

    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]

    if (dl != 0.) or (dm != 0.):
        frot = fringe_rotation(dl, dm, u, v, freq/freq_orig[0])
        for j in xrange(len1):
            for i in xrange(len0):
                for l in xrange(len3):    # iterate over pols
                    for k in xrange(len2):
                        data[i,j,k,l] = data[i,j,k,l] * frot[j,k]
    else:
        if verbose:
            print 'No phase rotation needed'

def calc_blarr(d):
    """ Helper function to make blarr a function instead of big list in d.
    ms and sdm format data have different bl orders.
    Could extend this to define useful subsets of baselines for (1) parallelization and (2) data weighting, and (3) data selection.
    """

    if d['dataformat'] == 'sdm':
        return n.array([ [d['ants'][i],d['ants'][j]] for j in range(d['nants']) for i in range(0,j) if ((d['ants'][i] not in d['excludeants']) and (d['ants'][j] not in d['excludeants']))])
    elif d['dataformat'] == 'ms':
        return n.array([[d['ants'][i],d['ants'][j]] for i in range(d['nants'])  for j in range(i+1, d['nants']) if ((d['ants'][i] not in d['excludeants']) and (d['ants'][j] not in d['excludeants']))])

cpdef n.ndarray[n.int_t, ndim=1] calc_delay(n.ndarray[float, ndim=1] freq, float inttime, float dm):
    """ Function to calculate delay for each channel in integrations.
    Takes freq array in GHz, inttime in s, and dm in pc/cm3.
    """

    cdef float freqref = freq[len(freq)-1]

    return n.round((4.2e-3 * dm * (1/(freq*freq) - 1/(freqref*freqref)))/inttime,0).astype(n.int16)

cpdef calc_resample(float chanwidth, float midfreq, float dm, float inttime):
    """ Function to calculate resmapling factor.
    freqs in GHz. inttime in s. returns intrachannel smearing by number of integrations.
    """

    return n.round(n.sqrt( (8.3e-3 * dm * chanwidth / midfreq**3)**2 + inttime**2)/inttime, 0).astype(n.int16)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, d, float dm, int verbose=0):
    """ dedisperse the data in place
    replaces data in place. dm algorithm on only accurate if moving "up" in dm space.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    cdef int len0
    cdef unsigned int len1
    cdef unsigned int len2
    cdef unsigned int len3
# making copy is slower, but returns actual roll of original data. without, we can only move "up" in dm space.
#    cdef n.ndarray[DTYPE_t, ndim=4] data1 = n.empty_like(data)

    # calc relative delay per channel. only shift minimally
    cdef n.ndarray[short, ndim=1] newdelay = calc_delay(d['freq_orig'][d['chans']], d['inttime'], dm)
    cdef n.ndarray[short, ndim=1] relativedelay = newdelay - d['datadelay']

    shape = n.shape(data)
    len0 = shape[0]
    len1 = shape[1]
    len2 = shape[2]
    len3 = shape[3]

#    print relativedelay
#    for shift in n.unique(relativedelay):
    for j in xrange(len1):
        for k in xrange(len2):
            shift = relativedelay[k]
            if shift > 0:
                for l in xrange(len3):
# following is cython shortcut for 'iprime = n.mod(i+shift, len0)'
                    for i in xrange(len0):
                        iprime = i+shift
#                    print i, iprime
                        if iprime >= 0 and iprime < len0:    # ignore edge cases
                            data[i,j,k,l] = data[iprime,j,k,l]
                        elif iprime >= len0:    # set nonsense shifted data to zero
                            data[i,j,k,l] = 0j
            elif shift < 0:
                print 'negative delay found. this dedispersion algorithm only works for positive delays. ignoring...'

# alternatives
#                            data1[i,j,k,l] = data[iprime,j,k,l]
#            data[:,:,indmin:indmax+1,:] = n.roll(data.take(range(indmin,indmax+1), axis=2), -1*shift, axis=0)

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

    # new delay values
    d['datadelay'] = newdelay
#    return data1        # if doing proper roll, need to return copy

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse_resample(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, n.ndarray[float, ndim=1] freq, float inttime, float dm, unsigned int resample, blr, int verbose=0):
    """ dedisperse the data and resample in place. only fraction of array is useful data.
    dm algorithm on only accurate if moving "up" in dm space.
    assumes unshifted data.
    only does resampling by dt. no dm resampling for now.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int r
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    # calc relative delay per channel. only shift minimally
    cdef n.ndarray[short, ndim=1] relativedelay = calc_delay(freq, inttime, dm)
    cdef unsigned int newlen0 = len0/resample

    for j in xrange(*blr):     # parallelized over blrange
        for l in xrange(len3):
            for k in xrange(len2):
                shift = relativedelay[k]
                for i in xrange(newlen0):
                    iprime = i*resample+shift
                    if iprime >= 0 and iprime < len0-(resample-1):    # if within bounds of unshifted data with resample stepping
                        data[i,j,k,l] = data[iprime,j,k,l]
                        if resample > 1:
                            for r in xrange(1,resample):
                                data[i,j,k,l] = data[i,j,k,l] + data[iprime+r,j,k,l]
                            data[i,j,k,l] = data[i,j,k,l]/resample
                    elif iprime >= len0-(resample):    # set nonsense shifted data to zero
                        data[i,j,k,l] = 0j

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse_par(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, n.ndarray[float, ndim=1] freq, float inttime, float dm, blr, int verbose=0):
    """ dedisperse the data in place. only fraction of array is useful data.
    dm algorithm only accurate if moving "up" in dm space.
    assumes unshifted data.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int r
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    # calc relative delay per channel. only shift minimally
    cdef n.ndarray[short, ndim=1] relativedelay = calc_delay(freq, inttime, dm)

    for j in xrange(*blr):     # parallelized over blrange
        for k in xrange(len2):
            shift = relativedelay[k]
            if shift > 0:
                for i in xrange(len0-shift):
                    iprime = i+shift
                    for l in xrange(len3):
                        data[i,j,k,l] = data[iprime,j,k,l]

    if verbose != 0:
        print 'Dedispersed for DM of %d' % dm

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef resample_par(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, n.ndarray[float, ndim=1] freq, float inttime, unsigned int resample, blr, int verbose=0):
    """ resample the data in place. only fraction of array is useful data.
    resample algorithm only accurate if moving "up" in dt space.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int r
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int newlen0 = len0/resample

    for j in xrange(*blr):     # parallelized over blrange
        for i in xrange(newlen0):
            iprime = i*resample
            for l in xrange(len3):
                for k in xrange(len2):
                    data[i,j,k,l] = data[iprime,j,k,l]
                    for r in xrange(1,resample):
                        data[i,j,k,l] = data[i,j,k,l] + data[iprime+r,j,k,l]
                    data[i,j,k,l] = data[i,j,k,l]/resample

    if verbose != 0:
        print 'Resampled by factor of %d' % resample

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse_resample2(n.ndarray[DTYPE_t, ndim=4, mode='c'] data, n.ndarray[float, ndim=1] freq, float inttime, float dm, unsigned int resample, int verbose=0):
    """ dedisperse the data and resample with fixed value in place. only fraction of array is useful data.
    dm algorithm on only accurate if moving "up" in dm space.
    assumes unshifted data.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int r
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef float chanwidth = freq[1] - freq[0]
    cdef float midfreq = freq[len(freq)/2]
    cdef unsigned int newlen0 = len0/resample

    if n.mod(len0, resample):
        print 'Warning: data length is not an integer number of resamples. Last int(s) will be lost.'

    # calc relative delay per channel. only shift minimally
    cdef n.ndarray[short, ndim=1] relativedelay = calc_delay(freq, inttime, dm)

    for j in xrange(len1):
        for l in xrange(len3):
            for k in xrange(len2):
                shift = relativedelay[k]
                for i in xrange(newlen0):
                    iprime = i*resample+shift
                    if iprime >= 0 and iprime < len0-(resample-1):    # if within bounds of unshifted data with resample stepping
                        data[i,j,k,l] = data[iprime,j,k,l]
                        if resample > 1:
                            for r in xrange(1,resample):
                                data[i,j,k,l] = data[i,j,k,l] + data[iprime+r,j,k,l]
                            data[i,j,k,l] = data[i,j,k,l]/resample
                    elif iprime >= len0-resample:    # set nonsense shifted data to zero
                        data[i,j,k,l] = 0j

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

cpdef meantsub(n.ndarray[DTYPE_t, ndim=4, mode='c'] datacal, blr):
    """ Subtract mean visibility, ignoring zeros
    """

    cdef unsigned int i, j, k, l
    sh = datacal.shape
    cdef unsigned int iterint = sh[0]
    cdef unsigned int nbl = sh[1]
    cdef unsigned int nchan = sh[2]
    cdef unsigned int npol = sh[3]
    cdef complex sum
    cdef unsigned int count = 0

    for j in xrange(*blr):
        for k in xrange(nchan):
            for l in xrange(npol):
                sum = 0.
                count = 0
                for i in xrange(iterint):
                    if datacal[i,j,k,l] != 0j:   # ignore zeros
                        sum += datacal[i,j,k,l]
                        count += 1
                if count:
                    for i in xrange(iterint):
                        if datacal[i,j,k,l] != 0j:   # ignore zeros
                            datacal[i,j,k,l] = datacal[i,j,k,l] - sum/count

cpdef dataflag(n.ndarray[DTYPE_t, ndim=4, mode='c'] datacal, n.ndarray[n.int_t, ndim=1] chans, unsigned int pol, d, sigma=4, mode='', convergence=0.2, tripfrac=0.4):
    """ Flagging function that can operate on pol/chan selections independently
    """

    cdef unsigned int i, j, 
    cdef unsigned int flagged = 0
    cdef unsigned int badpol
    cdef unsigned int badbl
    cdef unsigned int chan
    sh = datacal.shape
    cdef unsigned int iterint = sh[0]
    cdef unsigned int nbl = sh[1]
    cdef unsigned int nchan = sh[2]
    cdef unsigned int npol = sh[3]
    cdef n.ndarray[n.int_t, ndim=1] badbls
    cdef n.ndarray[n.int_t, ndim=1] badpols
    cdef float blstdmed, blstdstd

    flagged = 0
    if n.any(datacal[:,:,chans,pol]):

        if mode == 'blstd':
            blstd = datacal[:,:,chans,pol].std(axis=1)

            # iterate to good median and std values
            blstdmednew = n.ma.median(blstd)
            blstdstdnew = blstd.std()
            blstdstd = blstdstdnew*2
            while (blstdstd-blstdstdnew)/blstdstd > convergence:
                blstdstd = blstdstdnew
                blstdmed = blstdmednew
                blstd = n.ma.masked_where( blstd > blstdmed + sigma*blstdstd, blstd, copy=False)
                blstdmednew = n.ma.median(blstd)
                blstdstdnew = blstd.std()

            # flag blstd too high
            badint, badchan = n.where(blstd > blstdmednew + sigma*blstdstdnew)
            for badi in range(len(badint)):
#            for i in xrange(iterint):
#                for chan in xrange(len(chans)):
#                    if blstd.data[i,chan] > blstdmednew + sigma*blstdstdnew:     # then measure points to flag based on a third std threshold
                flagged += nbl
                for j in xrange(nbl):
                    datacal[badint[badi],j,chans[badchan[badi]],pol] = n.complex64(0j)
#                logger.info('%d, %d, %d, %d' % (badi, badint[badi], badchan[badi], chans[badchan[badi]]))

            summary='Blstd flagging for (chans %d-%d, pol %d), %.1f sigma: %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, 100.*flagged/datacal.size)

        elif mode == 'badchtslide':
            win = 10  # window to calculate median

            meanamp = n.abs(datacal[:,:,chans,pol]).mean(axis=1)
            spec = meanamp.mean(axis=0)
            lc = meanamp.mean(axis=1)

            # calc badch as deviation from median of window
            specmed = []
            for ch in range(len(spec)):
                rr = range(max(0, ch-win/2), min(len(spec), ch+win/2))
                rr.remove(ch)
                specmed.append(spec[ch] - n.median(spec[rr]))

            specmed = n.array(specmed)
            badch = n.where(specmed > sigma*specmed.std())[0]
            for chan in badch:
                flagged += iterint*nbl
                for i in xrange(iterint):
                    for j in xrange(nbl):
                        datacal[i,j,chan,pol] = n.complex64(0j)

            # calc badt as deviation from median of window
            lcmed = []
            for t in range(len(lc)):
                rr = range(max(0, t-win/2), min(len(lc), t+win/2))
                rr.remove(t)
                lcmed.append(lc[t] - n.median(lc[rr]))

            lcmed = n.array(lcmed)
            badt = n.where(lcmed > sigma*lcmed.std())[0]
            for i in badt:
                flagged += nchan*nbl
                for chan in xrange(len(chans)):
                    for j in xrange(nbl):
                        datacal[i,j,chans[chan],pol] = n.complex64(0j)

            summary='Bad chans/ints flagging for (chans %d-%d, pol %d), %1.f sigma: %d chans, %d ints, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badch), len(badt), 100.*flagged/datacal.size)

        elif mode == 'badcht':
            meanamp = n.abs(datacal[:,:,chans,pol]).mean(axis=1)

            # iterate to good median and std values
            meanampmednew = n.ma.median(meanamp)
            meanampstdnew = meanamp.std()
            meanampstd = meanampstdnew*2
            while (meanampstd-meanampstdnew)/meanampstd > convergence:
                meanampstd = meanampstdnew
                meanampmed = meanampmednew
                meanamp = n.ma.masked_where(meanamp > meanampmed + sigma*meanampstd, meanamp, copy=False)
                meanampmednew = n.ma.median(meanamp)
                meanampstdnew = meanamp.std()

            badch = chans[n.where( (meanamp.mean(axis=0) > meanampmednew + sigma*meanampstdnew) | (meanamp.mean(axis=0).mask==True) )[0]]
            badt = n.where( (meanamp.mean(axis=1) > meanampmednew + sigma*meanampstdnew) | (meanamp.mean(axis=1).mask==True) )[0]

            for chan in badch:
                flagged += iterint*nbl
                for i in xrange(iterint):
                    for j in xrange(nbl):
                        datacal[i,j,chan,pol] = n.complex64(0j)
            for i in badt:
                flagged += nchan*nbl
                for chan in xrange(len(chans)):
                    for j in xrange(nbl):
                        datacal[i,j,chans[chan],pol] = n.complex64(0j)

            summary='Bad chans/ints flagging for (chans %d-%d, pol %d), %1.f sigma: %d chans, %d ints, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badch), len(badt), 100.*flagged/datacal.size)

        elif mode == 'ring':
            spfft = n.abs(n.fft.ifft(datacal.mean(axis=0), axis=1))   # delay spectrum of mean data in time
            spfft = n.ma.masked_array(spfft, spfft = 0)
            badbls = n.where(spfft[:,len(chans)/2-1:len(chans)/2].mean(axis=1) > sigma*n.ma.median(spfft[:,1:], axis=1))[0]  # find bls with spectral power at max delay. ignore dc in case this is cal scan.
            if len(badbls) > tripfrac*nbl:    # if many bls affected, flag all
                print 'Ringing on %d/%d baselines. Flagging all data.' % (len(badbls), nbl)
                badbls = n.arange(nbl)

            for badbl in badbls:
               flagged += iterint*len(chans)
               for i in xrange(iterint):
                   for chan in chans:
                       datacal[i,badbl,chan,pol] = n.complex64(0j)

            summary='Ringing flagging for (chans %d-%d, pol %d) at %.1f sigma: %d/%d bls, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badbls), nbl, 100.*flagged/datacal.size)

        elif mode == 'badap':
            blarr = calc_blarr(d)
            bpa = n.abs(datacal[:,:,chans]).mean(axis=2).mean(axis=0)
            bpa_ant = n.array([ (bpa[n.where(n.any(blarr == i, axis=1))[0]]).mean(axis=0) for i in n.unique(blarr) ])
            bpa_ant = n.ma.masked_invalid(bpa_ant)
            ww = n.where(bpa_ant > n.ma.median(bpa_ant) + sigma * bpa_ant.std())
            badants = n.unique(blarr)[ww[0]]
            if len(badants):
                badbls = n.where(n.any(blarr == badants[0], axis=1))[0]   # initialize
                badpols = n.array([ww[1][0]]*len(badbls))
                for i in xrange(1, len(badants)):
                    newbadbls = n.where(n.any(blarr == badants[i], axis=1))[0]
                    badbls = n.concatenate( (badbls, newbadbls) )
                    badpols = n.concatenate( (badpols, [ww[1][i]]*len(newbadbls)) )
                for j in xrange(len(badbls)):
                    flagged += iterint*len(chans)
                    for i in xrange(iterint):
                        for chan in chans:
                            datacal[i,badbls[j],chan,badpols[j]] = n.complex64(0j)

            summary='Bad basepol flagging for chans %d-%d at %.1f sigma: ants/pols %s/%s, %3.2f %% of total flagged' % (chans[0], chans[-1], sigma, badants, ww[1], 100.*flagged/datacal.size)

        else:
            summary = 'Flagmode not recognized.'
    else:
        summary = 'Data already flagged for chans %d-%d, pol %d' % (chans[0], chans[-1], pol)

    return summary