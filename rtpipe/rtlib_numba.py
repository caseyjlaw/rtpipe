import numpy as np
from numba import jit, complex64, float32, int32, boolean
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# can choose between numpy and pyfftw
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    from numpy import fft


# must flatten?!
@jit 
def flag_calcmad(data):
    """ Calculate median absolute deviation of data array """

    absdev = np.abs(data - np.median(data))
    return np.median(absdev)


# @jit
# def flag_badcell(data, mad, sigma):
#     """ Flags peaks in data ch/t pixels """

#     (nints, nbl, nchan, npol) = data.shape
#     flags = np.zeros((nints, nbl, nchan, npol), dtype=bool)

#     for i in nints:
#         for j in nbl:
#             for k in nchan:
#                 for k in npol:
#                     if data[i,j,k] > sigma*mad:
#                         flags[i,j,k] = True

#     return flags


# @jit
# def flag_badch(data, mad, sigma):
#     """ Flags channels and integrations higher than sigma*mad """

#     meanamp = np.abs(data).mean(axis=1)

#     badch = np.where( (meanamp.mean(axis=0) > sigma*mad) | (meanamp.mean(axis=0).mask==True) )
#     badt = np.where( (meanamp.mean(axis=1) > sigma*mad) | (meanamp.mean(axis=1).mask==True) )

#     for chan in badch:
#         flagged += iterint*nbl
#         for i in xrange(iterint):
#             for j in xrange(nbl):
#                 datacal[i,j,chan,pol] = np.complex64(0j)
#     for i in badt:
#         flagged += nchan*nbl
#         for chan in xrange(len(chans)):
#             for j in xrange(nbl):
#                 datacal[i,j,chans[chan],pol] = np.complex64(0j)

#     summary='Bad chans/ints flagging for (chans %d-%d, pol %d), %1.f sigma: %d chans, %d ints, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badch), len(badt), 100.*flagged/datacal.size)



# @jit
# def dataflag(datacal, chans, pol, d, sigma = 4, mode = '', convergence = 0.2, tripfrac = 0.4):
#     """ Flagging function that can operate on pol/chan selections independently """

#     flagged = 0
#     (iterint, nbl, nchan, npol) = datacal.shape

#     if np.any(datacal[:,:,chans,pol]):

#         if mode == 'blstd':
#             blstd = datacal[:,:,chans,pol].std(axis=1)

#             # iterate to good median and std values
#             blstdmednew = np.ma.median(blstd)
#             blstdstdnew = blstd.std()
#             blstdstd = blstdstdnew*2
#             while (blstdstd-blstdstdnew)/blstdstd > convergence:
#                 blstdstd = blstdstdnew
#                 blstdmed = blstdmednew
#                 blstd = np.ma.masked_where( blstd > blstdmed + sigma*blstdstd, blstd, copy=False)
#                 blstdmednew = np.ma.median(blstd)
#                 blstdstdnew = blstd.std()

#             # flag blstd too high
#             badint, badchan = np.where(blstd > blstdmednew + sigma*blstdstdnew)
#             for badi in range(len(badint)):
# #            for i in xrange(iterint):
# #                for chan in xrange(len(chans)):
# #                    if blstd.data[i,chan] > blstdmednew + sigma*blstdstdnew:     # then measure points to flag based on a third std threshold
#                 flagged += nbl
#                 for j in xrange(nbl):
#                     datacal[badint[badi],j,chans[badchan[badi]],pol] = np.complex64(0j)
# #                logger.info('%d, %d, %d, %d' % (badi, badint[badi], badchan[badi], chans[badchan[badi]]))

#             summary='Blstd flagging for (chans %d-%d, pol %d), %.1f sigma: %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, 100.*flagged/datacal.size)

#         # elif mode == 'badchtslide':
#         #     win = 10  # window to calculate median

#         #     meanamp = np.abs(datacal[:,:,chans,pol]).mean(axis=1)
#         #     spec = meanamp.mean(axis=0)
#         #     lc = meanamp.mean(axis=1)

#         #     # calc badch as deviation from median of window
#         #     specmed = []
#         #     for ch in range(len(spec)):
#         #         rr = range(max(0, ch-win/2), min(len(spec), ch+win/2))
#         #         rr.remove(ch)
#         #         specmed.append(spec[ch] - np.median(spec[rr]))

#         #     specmed = np.array(specmed)
#         #     badch = np.where(specmed > sigma*specmed.std())[0]
#         #     for chan in badch:
#         #         flagged += iterint*nbl
#         #         for i in xrange(iterint):
#         #             for j in xrange(nbl):
#         #                 datacal[i,j,chan,pol] = np.complex64(0j)

#         #     # calc badt as deviation from median of window
#         #     lcmed = []
#         #     for t in range(len(lc)):
#         #         rr = range(max(0, t-win/2), min(len(lc), t+win/2))
#         #         rr.remove(t)
#         #         lcmed.append(lc[t] - np.median(lc[rr]))

#         #     lcmed = np.array(lcmed)
#         #     badt = np.where(lcmed > sigma*lcmed.std())[0]
#         #     for i in badt:
#         #         flagged += nchan*nbl
#         #         for chan in xrange(len(chans)):
#         #             for j in xrange(nbl):
#         #                 datacal[i,j,chans[chan],pol] = np.complex64(0j)

#         #     summary='Bad chans/ints flagging for (chans %d-%d, pol %d), %1.f sigma: %d chans, %d ints, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badch), len(badt), 100.*flagged/datacal.size)

#         elif mode == 'badcht':
#             meanamp = np.abs(datacal[:,:,chans,pol]).mean(axis=1)

#             # iterate to good median and std values
#             meanampmednew = np.ma.median(meanamp)
#             meanampstdnew = meanamp.std()
#             meanampstd = meanampstdnew*2
#             while (meanampstd-meanampstdnew)/meanampstd > convergence:
#                 meanampstd = meanampstdnew
#                 meanampmed = meanampmednew
#                 meanamp = np.ma.masked_where(meanamp > meanampmed + sigma*meanampstd, meanamp, copy=False)
#                 meanampmednew = np.ma.median(meanamp)
#                 meanampstdnew = meanamp.std()

#             badch = chans[np.where( (meanamp.mean(axis=0) > meanampmednew + sigma*meanampstdnew) | (meanamp.mean(axis=0).mask==True) )[0]]
#             badt = np.where( (meanamp.mean(axis=1) > meanampmednew + sigma*meanampstdnew) | (meanamp.mean(axis=1).mask==True) )[0]

#             for chan in badch:
#                 flagged += iterint*nbl
#                 for i in xrange(iterint):
#                     for j in xrange(nbl):
#                         datacal[i,j,chan,pol] = np.complex64(0j)
#             for i in badt:
#                 flagged += nchan*nbl
#                 for chan in xrange(len(chans)):
#                     for j in xrange(nbl):
#                         datacal[i,j,chans[chan],pol] = np.complex64(0j)

#             summary='Bad chans/ints flagging for (chans %d-%d, pol %d), %1.f sigma: %d chans, %d ints, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badch), len(badt), 100.*flagged/datacal.size)

#         elif mode == 'ring':
#             spfft = np.abs(np.fft.ifft(datacal.mean(axis=0), axis=1))   # delay spectrum of mean data in time
#             spfft = np.ma.masked_array(spfft, spfft = 0)
#             badbls = np.where(spfft[:,len(chans)/2-1:len(chans)/2].mean(axis=1) > sigma*np.ma.median(spfft[:,1:], axis=1))[0]  # find bls with spectral power at max delay. ignore dc in case this is cal scan.
#             if len(badbls) > tripfrac*nbl:    # if many bls affected, flag all
#                 print 'Ringing on %d/%d baselines. Flagging all data.' % (len(badbls), nbl)
#                 badbls = np.arange(nbl)

#             for badbl in badbls:
#                flagged += iterint*len(chans)
#                for i in xrange(iterint):
#                    for chan in chans:
#                        datacal[i,badbl,chan,pol] = np.complex64(0j)

#             summary='Ringing flagging for (chans %d-%d, pol %d) at %.1f sigma: %d/%d bls, %3.2f %% of total flagged' % (chans[0], chans[-1], pol, sigma, len(badbls), nbl, 100.*flagged/datacal.size)

#         else:
#             summary = 'Flagmode not recognized.'
#     else:
#         summary = 'Data already flagged for chans %d-%d, pol %d' % (chans[0], chans[-1], pol)

#     return summary



# @jit    
# def beamonefullxy(u, v, data, npixx, npixy, res):
#     """ Same as imgonefullxy, but returns dirty beam
#     Ignores uv points off the grid
#     flips xy gridding! im on visibility flux scale!
#     on flux scale (counts nonzero data)."""

#     # initial definitions
#     (len0, len1, len2) = np.shape(data)
#     nonzeros = 0
#     grid = np.zeros( (npixx,npixy), dtype='complex64')

#     # put uv data on grid
#     uu = np.round(u/res).astype(np.int)
#     vv = np.round(v/res).astype(np.int)

#     arr = pyfftw.n_byte_align_empty((npixx,npixy), 16, dtype='complex64')
#     ifft = pyfftw.builders.ifft2(arr, overwrite_input=True,
#                                  auto_align_input=True, auto_contiguous=True)

#     ok = np.logical_and(np.abs(uu) < npixx/2, np.abs(vv) < npixy/2)
#     uu = np.mod(uu, npixx)
#     vv = np.mod(vv, npixy)

#     # add uv data to grid
#     # or use np.add.at(x, i, y)?
#     for i in xrange(len0):
#         for j in xrange(len1):
#             if ok[i,j]:
#                 cellu = uu[i,j]
#                 cellv = vv[i,j]
#                 for p in xrange(len2):
#                     if data[i,j,p] != 0j:
#                         grid[cellu, cellv] = 1 + grid[cellu, cellv] 
#                         nonzeros = nonzeros + 1

#     # make images and filter based on threshold
#     arr[:] = grid[:]
#     im = ifft(arr).real*int(npixx*npixy)/float(nonzeros)
#     im = recenter(im, (npixx/2,npixy/2))

#     print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(npixx*npixy)/float(nonzeros))
#     print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*np.degrees(2./(npixx*res)), 3600*np.degrees(2./(npixy*res)), 3600*np.degrees(1./res))
#     return im


# @jit
# def imgonefullxy(u, v, data, npixx, npixy, uvres, verbose=1):
#     """Same as imgallfullxy, but one flux scaled image
#     Defines uvgrid filter before loop
#     flips xy gridding!"""

#     # initial definitions
#     (len0, len1, len2) = np.shape(data)
#     nonzeros = 0
#     grid = np.zeros((npixx,npixy), dtype='complex64')
#     arr = pyfftw.n_byte_align_empty((npixx,npixy), 16, dtype='complex64')

#     # put uv data on grid
#     uu = np.round(u/uvres).astype(np.int)
#     vv = np.round(v/uvres).astype(np.int)

#     ifft = pyfftw.builders.ifft2(arr, overwrite_input=True, auto_align_input=True, auto_contiguous=True)
    
#     ok = np.logical_and(np.abs(uu) < npixx/2, np.abs(vv) < npixy/2)
#     uu = np.mod(uu, npixx)
#     vv = np.mod(vv, npixy)

#     # add uv data to grid
#     # or use np.add.at(x, i, y)?
#     for i in xrange(len0):
#         for j in xrange(len1):
#             if ok[i,j]:
#                 cellu = uu[i,j]
#                 cellv = vv[i,j]
#                 for p in xrange(len2):
#                     grid[cellu, cellv] = data[i,j,p] + grid[cellu, cellv]
#                     if data[i,j,p] != 0j:
#                         nonzeros = nonzeros + 1

#     # make images and filter based on threshold
#     arr[:] = grid[:]
#     im = ifft(arr).real*int(npixx*npixy)
#     im = recenter(im, (npixx/2,npixy/2))
    
#     if nonzeros > 0:
#         im = im/float(nonzeros)
#         if verbose:
#             print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(npixx*npixy)/float(nonzeros))
#     else:
#         if verbose:
#             print 'Gridded %.3f of data. All zeros.' % (float(ok.sum())/ok.size)
#     if verbose:
#         print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*np.degrees(2./(npixx*uvres)), 3600*np.degrees(2./(npixy*uvres)), 3600*np.degrees(1./uvres))
#     return im


# @jit
# def imgallfullfilterxy(u, v, data, npixx, npixy, res, thresh):
#     """ Same as imgallfull, but returns both pos and neg candidates
#     Defines uvgrid filter before loop
#     flips xy gridding!"""

#     # initial definitions
#     (len0, len1, len2, len3) = np.shape(data)
#     grid = np.zeros((len0,npixx,npixy), dtype='complex64')
#     arr = pyfftw.n_byte_align_empty((npixx,npixy), 32, dtype='complex64')

#     # put uv data on grid
#     uu = np.round(u/res).astype(np.int)
#     vv = np.round(v/res).astype(np.int)

#     ifft = pyfftw.builders.ifft2(arr, overwrite_input=True,
#                                  auto_align_input=True, auto_contiguous=True,
#                                  planner_effort='FFTW_PATIENT')
    
#     ok = np.logical_and(np.abs(uu) < npixx/2,
#                         np.abs(vv) < npixy/2)
#     uu = np.mod(uu, npixx)
#     vv = np.mod(vv, npixy)

#     # add uv data to grid
#     # or use np.add.at(x, i, y)?
#     for i in xrange(len1):
#         for j in xrange(len2):
#             if ok[i,j]:
#                 cellu = uu[i,j]
#                 cellv = vv[i,j]
#                 for t in xrange(len0):
#                     for p in xrange(len3):
#                         grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

#     # make images and filter based on threshold
#     candints = []; candims = []; candsnrs = []
#     for t in xrange(len0):
#         arr[:] = grid[t]
#         im = ifft(arr).real

#         # find most extreme pixel
#         snrmax = im.max()/im.std()
#         snrmin = im.min()/im.std()
#         if snrmax >= abs(snrmin):
#             snr = snrmax
#         else:
#             snr = snrmin
#         if ( (abs(snr) > thresh) & np.any(data[t,:,len2/3:,:])):
#             candints.append(t)
#             candsnrs.append(snr)
#             candims.append(recenter(im, (npixx/2,npixy/2)))

# #    print 'Detected %d candidates with at least third the band.' % len(candints)
# #    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*np.degrees(2./(npixx*res)), 3600*np.degrees(2./(npixy*res)), 3600*np.degrees(1./res))
#     return candims,candsnrs,candints


# @jit
# def imgallfullfilterxyflux(u, v, data, npixx, npixy, res, thresh):
#     """ Same as imgallfull, but returns only candidates and rolls images
#     Defines uvgrid filter before loop
#     flips xy gridding!
#     counts nonzero data and properly normalizes fft to be on flux scale """

#     # initial definitions
#     (len0, len1, len2, len3) = np.shape(data)
#     nonzeros = 0
#     grid = np.zeros((len0,npixx,npixy), dtype='complex64')
#     arr = pyfftw.n_byte_align_empty((npixx,npixy), 16, dtype='complex64')

#     # put uv data on grid
#     uu = np.round(u/res).astype(np.int)
#     vv = np.round(v/res).astype(np.int)

#     ifft = pyfftw.builders.ifft2(arr, overwrite_input=True,
#                                  auto_align_input=True, auto_contiguous=True,
#                                  planner_effort='FFTW_PATIENT')
    
#     ok = np.logical_and(np.abs(uu) < npixx/2, np.abs(vv) < npixy/2)
#     uu = np.mod(uu, npixx)
#     vv = np.mod(vv, npixy)

#     # add uv data to grid
#     # or use np.add.at(x, i, y)?
#     for i in xrange(len1):
#         for j in xrange(len2):
#             if ok[i,j]:
#                 cellu = uu[i,j]
#                 cellv = vv[i,j]
#                 for t in xrange(len0):
#                     for p in xrange(len3):
#                         grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

#     # make images and filter based on threshold
#     candints = []; candims = []; candsnrs = []
#     for t in xrange(len0):
#         arr[:] = grid[t]
#         im = ifft(arr).real*int(npixx*npixy)

#         # find most extreme pixel
#         snrmax = im.max()/im.std()
#         snrmin = im.min()/im.std()
#         if snrmax >= abs(snrmin):
#             snr = snrmax
#         else:
#             snr = snrmin
#         if ( (abs(snr) > thresh) & np.any(data[t,:,len2/3:,:])):

#             # calculate number of nonzero vis to normalize fft
#             nonzeros = 0
#             for i in xrange(len1):
#                 for j in xrange(len2):
#                     if ok[i,j]:
#                         for p in xrange(len3):
#                             if data[t,i,j,p] != 0j:
#                                 nonzeros = nonzeros + 1

#             candints.append(t)
#             candsnrs.append(snr)
#             candims.append(recenter(im/float(nonzeros), (npixx/2,npixy/2)))

#     return candims,candsnrs,candints


# @jit
# def imgonefullw(u, v, data, npix, uvres, blsets, kers, verbose=1):
#     """ Same as imgallfullxy, but includes w term """

#     # initial definitions
#     (len0, len1, len2) = np.shape(data)
#     grid = np.zeros((npix,npix), dtype='complex64')
#     arr = pyfftw.n_byte_align_empty((npix,npix), 16, dtype='complex64')

#     # put uv data on grid
#     uu = np.round(u/uvres).astype(np.int)
#     vv = np.round(v/uvres).astype(np.int)

#     ifft = pyfftw.builders.ifft2(arr)
    
#     uu = np.mod(uu, npix)
#     vv = np.mod(vv, npix)

#     # add uv data to grid
#     for kernum in xrange(len(kers)):
#         bls = blsets[kernum]
#         ker = kers[kernum]
#         ksize = len(ker)
#         ok = np.logical_and(uu < (npix-ksize), vv < (npix-ksize))
#         for i in bls:
#             for j in xrange(len1):
#                 if ok[i,j]:
#                     cellu = uu[i,j]
#                     cellv = vv[i,j]
#                     for p in xrange(len2):
# # option 0) no w-term correction
# #                        grid[cellu, cellv] = data[i,j,p] + grid[cellu, cellv]  # no conv gridding
# #                        gridacc[cellu, cellv] = data[i,j,p] + gridacc[cellu, cellv]  # no conv gridding
# # option 1) conv gridding
#                          for keru in xrange(ksize):
#                              for kerv in xrange(ksize):
#                                  grid[cellu+keru-ksize/2, cellv+kerv-ksize/2] = ker[keru,kerv]*data[i,j,p] + grid[cellu+keru-ksize/2, cellv+kerv-ksize/2]
# # option 2) fourier kernel (not working)
# #        print 'kernum:', kernum
# #        arr[:] = gridacc[:]
# #        imacc = fft()
# #        arr[:] = (imacc*ker)[:]   
# #        grid[:] += ifft()[:]

#     # make images and filter based on threshold
#     arr[:] = grid[:]
#     im = ifft().real
#     im = recenter(im, (npix/2,npix/2))

#     if verbose:
#         print 'Pixel size %.1f\". Field size %.1f\"' % (3600*np.degrees(2./(npix*uvres)), 3600*np.degrees(1./uvres))
#     return im

# @jit
# def genuvkernels(w, wres, npix, uvres, thresh=0.99, oversample=2, ksize=0):
#     """ Generate uv kernels for w-term imaging """

#     npix = npix*oversample
#     uvres = uvres/oversample
#     lmker = pyfftw.n_byte_align_empty( (npix, npix), 16, dtype='complex64')

#     dofft = pyfftw.builders.fft2(lmker)

#     # set up w planes
#     sqrt_w = np.sqrt(np.abs(w)) * np.sign(w)
#     numw = np.ceil(1.1*(sqrt_w.max()-sqrt_w.min())/wres).astype(int)  # number of w bins (round up)
#     wgrid = np.linspace(sqrt_w.min()*1.05, sqrt_w.max()*1.05, numw)

#     # Grab a chunk of uvw's that grid w to same point.
#     uvkers = []; blsets = []
#     for i in range(len(wgrid)-1):
#         # get baselines in this wgrid bin
#         blw = np.where( (sqrt_w > wgrid[i]) & (sqrt_w <= wgrid[i+1]) )
#         blsets.append(blw[0])
#         avg_w = np.average(w[blw])
#         print 'w %.1f to %.1f: Added %d/%d baselines' % (np.sign(wgrid[i])*wgrid[i]**2, np.sign(wgrid[i+1])*wgrid[i+1]**2, len(blw[0]), len(w))

#         if len(blw[0]):
#             # get image extent
#             lmker[:] = get_lmkernel(npix, uvres, avg_w).astype(DTYPE)

#             # uv kernel from inv fft of lm kernel
#             im = dofft()
#             uvker = recenter(im, (npix/2,npix/2))

#             if ksize == 0:
#                 # keep uvker above a fraction (thresh) of peak amp
#                 largey, largex = np.where(np.abs(uvker) > thresh*np.abs(uvker).max())
#                 ksize = max(largey.max()-largey.min(), largex.max()-largex.min())                # take range of high values to define kernel size
#                 uvker = uvker[npix/2-ksize/2:npix/2+ksize/2+1, npix/2-ksize/2:npix/2+ksize/2+1]
#                 uvkers.append((uvker/uvker.sum()).astype(DTYPE))
#             else:
#                 uvker = uvker[npix/2-ksize/2:npix/2+ksize/2+1, npix/2-ksize/2:npix/2+ksize/2+1]
#                 uvkers.append((uvker/uvker.sum()).astype(DTYPE))
#         else:
#             uvkers.append([])

#     return blsets, uvkers


# @jit
# def genlmkernels(w, wres, npix, uvres):
#     """ Generate all lm kernels for given w distribution for w-term imaging. """

#     # set up w planes
#     sqrt_w = np.sqrt(np.abs(w)) * np.sign(w)
#     numw = np.ceil(1.1*(sqrt_w.max() - sqrt_w.min()) / wres).astype(int)  # number of w bins (round up)
#     wgrid = np.linspace(sqrt_w.min()*1.05, sqrt_w.max()*1.05, numw)

#     # Grab a chunk of uvw's that grid w to same point.
#     lmkers = []; blsets = []
#     for i in range(len(wgrid)-1):
#         # get baselines in this wgrid bin
#         blw = np.where( (sqrt_w > wgrid[i]) & (sqrt_w <= wgrid[i+1]) )
#         blsets.append(blw[0])
#         avg_w = np.average(w[blw])
#         print 'w %.1f to %.1f: Added %d/%d baselines' % (np.sign(wgrid[i])*wgrid[i]**2, np.sign(wgrid[i+1])*wgrid[i+1]**2, len(blw[0]), len(w))

#         if len(blw[0]):
#             lmkers.append(get_lmkernel(npix, uvres, avg_w).astype(DTYPE))
#         else:
#             lmkers.append([])

#     return blsets, lmkers


# @jit
# def get_lmkernel(npix, res, avg_w):
#     """ Generate single lm kernel """

#     l, m = get_lm(npix, res)
#     sqrtn = np.sqrt(1 - l**2 - m**2).astype(np.float32)
#     G = np.exp(-2*np.pi*1j*avg_w*(sqrtn - 1))
#     G = G.filled(0)
#     # Unscramble difference between fft(fft(G)) and G
#     G[1:] = np.flipud(G[1:]).copy()
#     G[:,1:] = np.fliplr(G[:,1:]).copy()
#     return G / G.size


# @jit
# def get_lm(npix, res, center=(0,0)):
#     """ Calculate l,m map """
#     m,l = np.indices((npix,npix))
#     l,m = np.where(l > npix/2, npix-l, -l), np.where(m > npix/2, m-npix, m)
#     l,m = l.astype(np.float32)/npix/res, m.astype(np.float32)/npix/res
#     mask = np.where(l**2 + m**2 >= 1, 1, 0)
#     l,m = np.ma.array(l, mask=mask), np.ma.array(m, mask=mask)
#     return recenter(l, center), recenter(m, center)


# @jit
# def recenter(a, c):
#     """ Recenter image a to pixel given by c  """

#     s = a.shape
#     c = (c[0] % s[0], c[1] % s[1])
#     if np.ma.isMA(a):
#         a1 = np.ma.concatenate([a[c[0]:], a[:c[0]]], axis=0)
#         a2 = np.ma.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
#     else:
#         a1 = np.concatenate([a[c[0]:], a[:c[0]]], axis=0)
#         a2 = np.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
#     return a2


# @jit
# def sigma_clip(arr, sigma=3):
#     """ Function takes 1d array of values and returns the sigma-clipped min and max scaled by value "sigma". """

#     assert arr.dtype == np.float32

#     cliparr = np.arange(len(arr))
#     mean = 0
#     std = 0

#     arr = np.append(arr,[np.float32(1)])    # append superfluous item to trigger loop
#     while len(cliparr) != len(arr):
#         arr = arr[cliparr]
#         mean = arr.mean()
#         std = arr.std()
#         cliparr = np.where((arr < mean + sigma*std) & (arr > mean - sigma*std) & (arr != 0) )[0]
# #        print 'Clipping %d from array of length %d' % (len(arr) - len(cliparr), len(arr))
#     return mean - sigma*std, mean + sigma*std


# @jit
# def make_triples(d):
#     """ Calculates and returns data indexes (i,j,k) for all closed triples. """

#     ants = d['ants']
#     nants = d['nants']
#     blarr = calc_blarr(d)

#     triples = np.zeros((nants*(nants-1)*(nants-2)/6, 3), dtype='int')

#     # first make triples indexes in antenna numbering
#     anttrips = []
#     for i in ants:
#         for j in ants[list(ants).index(i)+1:]:
#             for k in ants[list(ants).index(j)+1:]:
#                 anttrips.append([i,j,k])
                
#     # next return data indexes for triples
#     for t in xrange(len(anttrips)):
#         ant1 = anttrips[t][0]
#         ant2 = anttrips[t][1]
#         ant3 = anttrips[t][2]
#         try:
#             bl1 = np.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant2) )[0][0]
#             bl2 = np.where( (blarr[:,0] == ant2) & (blarr[:,1] == ant3) )[0][0]
#             bl3 = np.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant3) )[0][0]
#             triples[t,0] = bl1
#             triples[t,1] = bl2
#             triples[t,2] = bl3
#         except IndexError:
#             continue

#     return triples


# @jit
# def fringe_rotation(dl, dm, u, v, freq): 
#     """ Calculate fringe rotation term for change in l,m and given u,v,freq vectors """

#     return np.exp(-2j*3.1415*(dl*np.outer(u,freq) + dm*np.outer(v,freq)))


# @jit
# def phaseshift(data, d, l1, m1, u, v, verbose=0):
#     """ Shift phase center to (l1, m1).
#     Assumes single uv over all times in data. Reasonable for up to a second or so of data. """

#     freq = d['freq_orig'][d['chans']]
#     freq_orig = d['freq_orig']
#     dl = l1 - d['l0']
#     dm = m1 - d['m0']
#     (len0, len1, len2, len3) = np.shape(data)

#     if (dl != 0.) or (dm != 0.):
#         frot = fringe_rotation(dl, dm, u[len0/2], v[len0/2], freq/freq_orig[0])
#         for j in xrange(len1):
#             for i in xrange(len0):
#                 for l in xrange(len3):    # iterate over pols
#                     for k in xrange(len2):
#                         data[i,j,k,l] = data[i,j,k,l] * frot[j,k]
#     else:
#         if verbose:
#             print 'No phase rotation needed'

#     d['l0'] = l1
#     d['m0'] = m1


# @jit
# def phaseshift_threaded(data, d, l1, m1, u, v, verbose=0):
#     """ Shift phase center to (l1, m1).
#     Assumes single uv over all times in data. Reasonable for up to a second or so of data. """

#     freq = d['freq_orig'][d['chans']]
#     freq_orig = d['freq_orig']
#     dl = l1 - d['l0']
#     dm = m1 - d['m0']

#     (len0, len1, len2, len3) = np.shape(data)

#     if (dl != 0.) or (dm != 0.):
#         frot = fringe_rotation(dl, dm, u, v, freq/freq_orig[0])
#         for j in xrange(len1):
#             for i in xrange(len0):
#                 for l in xrange(len3):    # iterate over pols
#                     for k in xrange(len2):
#                         data[i,j,k,l] = data[i,j,k,l] * frot[j,k]
#     else:
#         if verbose:
#             print 'No phase rotation needed'


# @jit
# def calc_blarr(d):
#     """ Helper function to make blarr a function instead of big list in d.
#     ms and sdm format data have different bl orders.
#     Could extend this to define useful subsets of baselines for (1) parallelization and (2) data weighting, and (3) data selection. """

#     if d['dataformat'] == 'sdm':
#         return np.array([ [d['ants'][i],d['ants'][j]] for j in range(d['nants']) for i in range(0,j) if ((d['ants'][i] not in d['excludeants']) and (d['ants'][j] not in d['excludeants']))])
#     elif d['dataformat'] == 'ms':
#         return np.array([[d['ants'][i],d['ants'][j]] for i in range(d['nants'])  for j in range(i+1, d['nants']) if ((d['ants'][i] not in d['excludeants']) and (d['ants'][j] not in d['excludeants']))])


# @jit
# def calc_delay(freq, inttime, dm):
#     """ Function to calculate delay for each channel in integrations.
#     Takes freq array in GHz, inttime in s, and dm in pc/cm3. """

#     freqref = freq[len(freq)-1]

#     return np.round((4.2e-3 * dm * (1/(freq*freq) - 1/(freqref*freqref)))/inttime,0).astype(np.int16)


# @jit
# def calc_resample(chanwidth, midfreq, dm, inttime):
#     """ Function to calculate resmapling factor.
#     freqs in GHz. inttime in s. returns intrachannel smearing by number of integrations. """

#     return np.round(np.sqrt( (8.3e-3 * dm * chanwidth / midfreq**3)**2 + inttime**2)/inttime, 0).astype(np.int16)


# @jit
# def dedisperse(data, d, dm, verbose=0):
#     """ dedisperse the data in place
#     replaces data in place. dm algorithm on only accurate if moving "up" in dm space. """

# # making copy is slower, but returns actual roll of original data. without, we can only move "up" in dm space.
# #    cdef np.ndarray[DTYPE_t, ndim=4] data1 = np.empty_like(data)

#     # calc relative delay per channel. only shift minimally
#     newdelay = calc_delay(d['freq_orig'][d['chans']], d['inttime'], dm)
#     relativedelay = newdelay - d['datadelay']

#     (len0, len1, len2, len3) = np.shape(data)

# #    print relativedelay
# #    for shift in np.unique(relativedelay):
#     for j in xrange(len1):
#         for k in xrange(len2):
#             shift = relativedelay[k]
#             if shift > 0:
#                 for l in xrange(len3):
# # following is cython shortcut for 'iprime = np.mod(i+shift, len0)'
#                     for i in xrange(len0):
#                         iprime = i+shift
# #                    print i, iprime
#                         if iprime >= 0 and iprime < len0:    # ignore edge cases
#                             data[i,j,k,l] = data[iprime,j,k,l]
#                         elif iprime >= len0:    # set nonsense shifted data to zero
#                             data[i,j,k,l] = 0j
#             elif shift < 0:
#                 print 'negative delay found. this dedispersion algorithm only works for positive delays. ignoring...'

# # alternatives
# #                            data1[i,j,k,l] = data[iprime,j,k,l]
# #            data[:,:,indmin:indmax+1,:] = np.roll(data.take(range(indmin,indmax+1), axis=2), -1*shift, axis=0)

#     if verbose != 0:
#         print 'Dedispersed for DM=%d' % dm

#     # new delay values
#     d['datadelay'] = newdelay
# #    return data1        # if doing proper roll, need to return copy


# @jit
# def dedisperse_resample(data, freq, inttime, dm, resample, blr, verbose=0):
#     """ dedisperse the data and resample in place. only fraction of array is useful data.
#     dm algorithm on only accurate if moving "up" in dm space.
#     assumes unshifted data.
#     only does resampling by dt. no dm resampling for now. """

#     (len0, len1, len2, len3) = np.shape(data)

#     # calc relative delay per channel. only shift minimally
#     relativedelay = calc_delay(freq, inttime, dm)
#     newlen0 = len0/resample

#     for j in xrange(*blr):     # parallelized over blrange
#         for l in xrange(len3):
#             for k in xrange(len2):
#                 shift = relativedelay[k]
#                 for i in xrange(newlen0):
#                     iprime = i*resample+shift
#                     if iprime >= 0 and iprime < len0-(resample-1):    # if within bounds of unshifted data with resample stepping
#                         data[i,j,k,l] = data[iprime,j,k,l]
#                         if resample > 1:
#                             for r in xrange(1,resample):
#                                 data[i,j,k,l] = data[i,j,k,l] + data[iprime+r,j,k,l]
#                             data[i,j,k,l] = data[i,j,k,l]/resample
#                     elif iprime >= len0-(resample):    # set nonsense shifted data to zero
#                         data[i,j,k,l] = 0j

#     if verbose != 0:
#         print 'Dedispersed for DM=%d' % dm


# @jit
# def dedisperse_par(data, freq, inttime, dm, blr, verbose=0):
#     """ dedisperse the data in place. only fraction of array is useful data.
#     dm algorithm only accurate if moving "up" in dm space.
#     assumes unshifted data. """

#     (len0, len1, len2, len3) = np.shape(data)

#     # calc relative delay per channel. only shift minimally
#     relativedelay = calc_delay(freq, inttime, dm)

#     for j in xrange(*blr):     # parallelized over blrange
#         for k in xrange(len2):
#             shift = relativedelay[k]
#             if shift > 0:
#                 for i in xrange(len0-shift):
#                     iprime = i+shift
#                     for l in xrange(len3):
#                         data[i,j,k,l] = data[iprime,j,k,l]

#     if verbose != 0:
#         print 'Dedispersed for DM of %d' % dm


# @jit
# def resample_par(data, freq, inttime, resample, blr, verbose=0):
#     """ resample the data in place. only fraction of array is useful data.
#     resample algorithm only accurate if moving "up" in dt space. """

#     (len0, len1, len2, len3) = np.shape(data)
#     newlen0 = len0/resample

#     for j in xrange(*blr):     # parallelized over blrange
#         for i in xrange(newlen0):
#             iprime = i*resample
#             for l in xrange(len3):
#                 for k in xrange(len2):
#                     data[i,j,k,l] = data[iprime,j,k,l]
#                     for r in xrange(1,resample):
#                         data[i,j,k,l] = data[i,j,k,l] + data[iprime+r,j,k,l]
#                     data[i,j,k,l] = data[i,j,k,l]/resample

#     if verbose != 0:
#         print 'Resampled by factor of %d' % resample


# @jit
# def dedisperse_resample2(data, freq, inttime, dm, resample, verbose=0):
#     """ dedisperse the data and resample with fixed value in place. only fraction of array is useful data.
#     dm algorithm on only accurate if moving "up" in dm space.
#     assumes unshifted data. """

#     (len0, len1, len2, len3) = np.shape(data)
#     chanwidth = freq[1] - freq[0]
#     midfreq = freq[len(freq)/2]
#     newlen0 = len0/resample

#     if np.mod(len0, resample):
#         print 'Warning: data length is not an integer number of resamples. Last int(s) will be lost.'

#     # calc relative delay per channel. only shift minimally
#     relativedelay = calc_delay(freq, inttime, dm)

#     for j in xrange(len1):
#         for l in xrange(len3):
#             for k in xrange(len2):
#                 shift = relativedelay[k]
#                 for i in xrange(newlen0):
#                     iprime = i*resample+shift
#                     if iprime >= 0 and iprime < len0-(resample-1):    # if within bounds of unshifted data with resample stepping
#                         data[i,j,k,l] = data[iprime,j,k,l]
#                         if resample > 1:
#                             for r in xrange(1,resample):
#                                 data[i,j,k,l] = data[i,j,k,l] + data[iprime+r,j,k,l]
#                             data[i,j,k,l] = data[i,j,k,l]/resample
#                     elif iprime >= len0-resample:    # set nonsense shifted data to zero
#                         data[i,j,k,l] = 0j

#     if verbose != 0:
#         print 'Dedispersed for DM=%d' % dm


# @jit
# def meantsub(datacal, blr):
#     """ Subtract mean visibility, ignoring zeros """

#     (iterint, nbl, nchan, npol) = datacal.shape
#     count = 0

#     for j in xrange(*blr):
#         for k in xrange(nchan):
#             for l in xrange(npol):
#                 sum = 0.
#                 count = 0
#                 for i in xrange(iterint):
#                     if datacal[i,j,k,l] != 0j:   # ignore zeros
#                         sum += datacal[i,j,k,l]
#                         count += 1
#                 if count:
#                     for i in xrange(iterint):
#                         if datacal[i,j,k,l] != 0j:   # ignore zeros
#                             datacal[i,j,k,l] = datacal[i,j,k,l] - sum/count




