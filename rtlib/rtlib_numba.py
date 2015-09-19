import pyfftw
import numpy as n
import numba

@numba.jit
def imgallfullfilterxy(u, v, data, npixx, npixy, res, thresh):
    # Same as imgallfull, but returns both pos and neg candidates
    # Defines uvgrid filter before loop
    # flips xy gridding!

    # initial definitions
    shape = n.shape(data)
    len0 = shape[0]
    len1 = shape[1]
    len2 = shape[2]
    len3 = shape[3]
    grid = n.zeros((len0,npixx,npixy), dtype='complex64')
    arr = pyfftw.n_byte_align_empty((npixx,npixy), 32, dtype='complex64')

    # put uv data on grid
    uu = n.round(u/res).astype(n.int)
    vv = n.round(v/res).astype(n.int)

    ifft = pyfftw.builders.ifft2(arr, overwrite_input=True, auto_align_input=True, auto_contiguous=True)
    
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
