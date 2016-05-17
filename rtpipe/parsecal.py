import numpy as n
import os, glob, sys
try:
    import casautil
except ImportError:
    import pwkit.environments.casa.util as casautil
import logging, logging.config

# set up
tb = casautil.tools.table()

class casa_sol():
    """ Container for CASA caltable(s).
    Provides tools for applying to data of shape (nints, nbl, nch, npol).
    Initialize class based on input file(s) and selection criteria.
    Optional flux scale gain file can be given. Should be gain file applied to source with setjy applied.
    """

    def __init__(self, gainfile, flagants=True):
        """ Initialize with a table of CASA gain solutions. Can later add BP.
        """
        
        # custom log file
        self.logger = logging.getLogger(__name__)
        if os.path.exists(gainfile):
            self.parsegain(gainfile)
            self.flagants = flagants
        else:
            self.logger.warn('Gainfile not found.')
            raise IOError

    def parsegain(self, gainfile):
        """Takes .g1 CASA cal table and places values in numpy arrays.
        """

        tb.open(gainfile)
        mjd = tb.getcol('TIME')/(24*3600)     # mjd days, as for telcal
        field = tb.getcol('FIELD_ID')
        spw = tb.getcol('SPECTRAL_WINDOW_ID')
        gain = tb.getcol('CPARAM')    # dimensions of (npol, 1?, ntimes*nants)
        snr = tb.getcol('SNR')
        flagged = tb.getcol('FLAG')
        tb.close()
        tb.open(os.path.join(gainfile, 'ANTENNA'))
        antname = tb.getcol('NAME')   # ant number in order written to gain file
        antnum = n.array([int(antname[i][2:]) for i in range(len(antname))])
        tb.close()
        tb.open(os.path.join(gainfile, 'FIELD'))
        caldir = tb.getcol('PHASE_DIR')   # shape = (2 dirs, 1?, n sources)
        calra = n.mod(caldir[0,0,:], 2*n.pi)
        caldec = caldir[1,0,:]
        self.radec = zip(calra, caldec)

        tb.close()

        # # need to find parent data MS to get some metadata
        # mslist = glob.glob(gainfile[:-3] + '*.ms')
        # try:
        #     msfile = mslist[0]
        #     self.logger.info('Found parent data MS %s' % msfile)
        # except IndexError:
        #     self.logger.warn('Could not find parent data MS for metadata...')

        # tb.open(msfile + '/ANTENNA')
        # antname = tb.getcol('NAME')      # one name per ant
        # tb.close()
        # tb.open(msfile + '/SPECTRAL_WINDOW')
        # reffreq = 1e-6*(tb.getcol('REF_FREQUENCY')+tb.getcol('TOTAL_BANDWIDTH')/2)   # similar to telcal "skyfreq"
        # specname = tb.getcol('NAME')
        # tb.close()
        # tb.open(msfile + '/SOURCE')
        # source = [name for name in tb.getcol('NAME') if 'J' in name][0]          # should return single cal name **hack**
        # tb.close()
        # nsol = len(gain[0,0])

        # ifid0R = specname[0][7] + '-' + specname[0][8]       # one value
        # ifid0L = specname[0][9] + '-' + specname[0][10]       # one value
        # ifid1R = specname[1][7] + '-' + specname[1][8]       # one value
        # ifid1L = specname[1][9] + '-' + specname[1][10]       # one value

        # # paste R,L end to end, so first loop over time, then spw, then pol
        # mjd = n.concatenate( (time, time), axis=0)
        # ifid = [ifid0R]*(nsol/2) + [ifid1R]*(nsol/2) + [ifid0L]*(nsol/2) + [ifid1L]*(nsol/2)   # first quarter is spw0,pol0, then spw1,pol0, ...
        # skyfreq = n.concatenate( (reffreq[0]*n.ones(nsol/2), reffreq[1]*n.ones(nsol/2), reffreq[0]*n.ones(nsol/2), reffreq[1]*n.ones(nsol/2)), axis=0)
        # gain = n.concatenate( (gain[0,0],gain[1,0]), axis=0)
        # amp = n.abs(gain)
        # phase = n.degrees(n.angle(gain))
        # source = [source]*nsol*2
        # flagged = n.concatenate( (flag[0,0],flag[1,0]), axis=0)
                   
        nants = len(n.unique(antnum))
        nspw = len(n.unique(spw))
        self.spwlist = n.unique(spw)
        npol = len(gain)

        # merge times less than some threshold
        nsol = 0
        newmjd = [n.unique(mjd)[0]]
        uniquefield = [field[n.where(newmjd[0] == mjd)][0]]

        skip = []
        for i in range(1, len(n.unique(mjd))):
            if 24*3600*(n.unique(mjd)[i] - n.unique(mjd)[i-1]) < 30.:
                skip.append(n.unique(mjd)[i])
                continue
            else:
                newmjd.append(n.unique(mjd)[i])
                uniquefield.append(field[n.where(n.unique(mjd)[i] == mjd)[0][0]])
        
        self.uniquemjd = n.array(newmjd)
        self.uniquefield = n.array(uniquefield)
        nsol = len(self.uniquemjd)

        self.logger.info('Parsed gain table solutions for %d solutions (skipping %d), %d ants, %d spw, and %d pols' % (nsol, len(skip), nants, nspw, npol))
        self.logger.info('Unique solution fields/times: %s' % str(zip(self.uniquefield, self.uniquemjd)))

        self.gain = n.zeros( (nsol, nants, nspw, npol), dtype='complex' )
        flags = n.zeros( (nsol, nants, nspw, npol), dtype='complex' )
        for sol in range(nsol):
            for ant in range(nants):
                for spw in range(nspw):
                    for pol in range(npol):
                        self.gain[sol, ant, spw, pol] = gain[pol,0,spw*nsol*nants+sol*nants+ant]
                        flags[sol, ant, spw, pol] = flagged[pol,0,spw*nsol*nants+sol*nants+ant]
        self.gain = n.ma.masked_array(self.gain, flags)

#        gain = n.concatenate( (n.concatenate( (gain[0,0,:nants*nsol].reshape(nsol,nants,1,1), gain[1,0,:nants*nsol].reshape(nsol,nants,1,1)), axis=3), n.concatenate( (gain[0,0,nants*nsol:].reshape(nsol,nants,1,1), gain[1,0,nants*nsol:].reshape(nsol,nants,1,1)), axis=3)), axis=2)
#        flagged = n.concatenate( (n.concatenate( (flagged[0,0,:nants*nsol].reshape(nsol,nants,1,1), flagged[1,0,:nants*nsol].reshape(nsol,nants,1,1)), axis=3), n.concatenate( (flagged[0,0,nants*nsol:].reshape(nsol,nants,1,1), flagged[1,0,nants*nsol:].reshape(nsol,nants,1,1)), axis=3)), axis=2)
#        self.gain = n.ma.masked_array(gain, flagged == True)        

        self.mjd = n.array(mjd); self.antnum = antnum

        # make another version of ants array
#        self.antnum = n.concatenate( (antnum, antnum), axis=0)
#        self.amp = n.array(amp); self.phase = n.array(phase)
#        self.antname = n.concatenate( (antname[antnum], antname[antnum]), axis=0)
#        self.complete = n.arange(len(self.mjd))

        # for consistency with telcal
        #self.ifid = n.array(ifid); self.skyfreq = n.array(skyfreq); self.source = n.array(source)

    def parsebp(self, bpfile, debug=False):
        """ Takes bp CASA cal table and places values in numpy arrays.
        Assumes two or fewer spw. :\
        Assumes one bp solution per file.
        """

        # bandpass. taking liberally from Corder et al's analysisutilities
        ([polyMode, polyType, nPolyAmp, nPolyPhase, scaleFactor, nRows, nSpws, nUniqueTimesBP, uniqueTimesBP,
          nPolarizations, frequencyLimits, increments, frequenciesGHz, polynomialPhase,
          polynomialAmplitude, timesBP, antennasBP, cal_desc_idBP, spwBP]) = openBpolyFile(bpfile, debug)

        # index iterates over antennas, then times/sources (solution sets). each index has 2x npoly, which are 2 pols
        polynomialAmplitude = n.array(polynomialAmplitude); polynomialPhase = n.array(polynomialPhase)
        polynomialAmplitude[:,0] = 0.; polynomialAmplitude[:,nPolyAmp] = 0.
        polynomialPhase[:,0] = 0.; polynomialPhase[nPolyPhase] = 0.
        ampSolR, ampSolL = calcChebyshev(polynomialAmplitude, frequencyLimits, n.array(frequenciesGHz)*1e+9)
        phaseSolR, phaseSolL = calcChebyshev(polynomialPhase, frequencyLimits, n.array(frequenciesGHz)*1e+9)

        nants = len(n.unique(antennasBP))
        self.bptimes = n.array(timesBP)
        ptsperspec = 1000
        npol = 2
        self.logger.info('Parsed bp solutions for %d solutions, %d ants, %d spw, and %d pols' % (nUniqueTimesBP, nants, nSpws, nPolarizations))
        self.bandpass = n.zeros( (nants, nSpws*ptsperspec, npol), dtype='complex')
        for spw in range(nSpws):
            ampSolR[spw*nants:(spw+1)*nants] += 1 - ampSolR[spw*nants:(spw+1)*nants].mean()     # renormalize mean over ants (per spw) == 1
            ampSolL[spw*nants:(spw+1)*nants] += 1 - ampSolL[spw*nants:(spw+1)*nants].mean()
            for ant in range(nants):
                self.bandpass[ant, spw*ptsperspec:(spw+1)*ptsperspec, 0] = ampSolR[ant+spw*nants] * n.exp(1j*phaseSolR[ant+spw*nants])
                self.bandpass[ant, spw*ptsperspec:(spw+1)*ptsperspec, 1] = ampSolL[ant+spw*nants] * n.exp(1j*phaseSolL[ant+spw*nants])

        self.bpfreq = n.zeros( (nSpws*ptsperspec) )
        for spw in range(nSpws):
            self.bpfreq[spw*ptsperspec:(spw+1)*ptsperspec] = 1e9 * frequenciesGHz[nants*spw]

#        bpSolR0 = ampSolR[:nants] * n.exp(1j*phaseSolR[:nants])
#        bpSolR1 = ampSolR[nants:] * n.exp(1j*phaseSolR[nants:])
#        bpSolL0 = ampSolL[:nants] * n.exp(1j*phaseSolL[:nants])
#        bpSolL1 = ampSolL[nants:] * n.exp(1j*phaseSolL[nants:])

        # structure close to tpipe data structure (nant, freq, pol). note that freq is oversampled to 1000 bins.
#        self.bandpass = n.concatenate( (n.concatenate( (bpSolR0[:,:,None], bpSolR1[:,:,None]), axis=1), n.concatenate( (bpSolL0[:,:,None], bpSolL1[:,:,None]), axis=1)), axis=2)
#        self.bpfreq = 1e9*n.concatenate( (frequenciesGHz[0], frequenciesGHz[nants]), axis=0)    # freq values at bp bins
#        self.logger.info('Parsed bp table solutions for %d solutions, %d ants, %d spw, and %d pols' % (nUniqueTimesBP, nants, nSpws, nPolarizations))

    def set_selection(self, time, freqs, blarr, calname='', radec=(), dist=1, spwind=[], pols=['XX','YY']):
        """ Set select parameter that defines time, spw, and pol solutions to apply.
        time defines the time to find solutions near in mjd.
        freqs defines frequencies to select bandpass solution
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        radec (radian tuple) and dist (deg) define optional location of source for filtering solutions.
        spwind is list of indices to be used (e.g., [0,2,4,10])
        pols is from d['pols'] (e.g., ['RR']). single or dual parallel allowed.
        calname not used. here for uniformity with telcal_sol.
        """

        self.spwind = spwind
        if calname:
            self.logger.warn('calname option not used for casa_sol. Applied based on radec.')

        # define pol index
        if 'X' in ''.join(pols) or 'Y' in ''.join(pols):
            polord = ['XX', 'YY']
        elif 'R' in ''.join(pols) or 'L' in ''.join(pols):
            polord = ['RR', 'LL']
        self.polind = [polord.index(pol) for pol in pols]

        self.ant1ind = [n.where(ant1 == n.unique(blarr))[0][0] for (ant1,ant2) in blarr]
        self.ant2ind = [n.where(ant2 == n.unique(blarr))[0][0] for (ant1,ant2) in blarr]

        # select by smallest time distance for source within some angular region of target
        if radec:
            ra, dec = radec
            calra = n.array(self.radec)[:,0]
            caldec = n.array(self.radec)[:,1]
            fields = n.where( (n.abs(calra - ra) < n.radians(dist)) & (n.abs(caldec - dec) < n.radians(dist)) )[0]
            if len(fields) == 0:
                self.logger.warn('Warning: no close calibrator found. Removing radec restriction.')
                fields = n.unique(self.uniquefield)
        else:
            fields = n.unique(self.uniquefield)

        sel = []
        for field in fields:
            sel += list(n.where(field == self.uniquefield)[0])
        mjddist = n.abs(time - self.uniquemjd[sel])
        closestgain = n.where(mjddist == mjddist.min())[0][0]

        self.logger.info('Using gain solution for field %d at MJD %.5f, separated by %d min ' % (self.uniquefield[n.where(self.uniquemjd == self.uniquemjd[sel][closestgain])], self.uniquemjd[closestgain], mjddist[closestgain]*24*60))
        self.gain = self.gain.take(self.spwind, axis=2).take(self.polind, axis=3)[closestgain]

        if hasattr(self, 'bandpass'):
            bins = [n.where(n.min(n.abs(self.bpfreq-selfreq)) == n.abs(self.bpfreq-selfreq))[0][0] for selfreq in freqs]
            self.bandpass = self.bandpass.take(bins, axis=1).take(self.polind, axis=2)
            self.freqs = freqs
            self.logger.debug('Using bandpass at BP bins (1000 bins per spw): %s', str(bins))

    def calc_flag(self, sig=3.0):
        """ Calculates antennas to flag, based on bad gain and bp solutions.
        """
 
        if len(self.gain.shape) == 4:
            gamp = n.abs(self.gain).mean(axis=0)   # mean gain amp for each ant over time
        elif len(self.gain.shape) == 3:
            gamp = n.abs(self.gain)   # gain amp for selected time

#        badgain = n.where(gamp < gamp.mean() - sig*gamp.std())
        badgain = n.where( (gamp < gamp.mean() - sig*gamp.std()) | gamp.mask)
        self.logger.info('Flagging low/bad gains for ant/spw/pol: %s %s %s' % (str(self.antnum[badgain[0]]), str(badgain[1]), str(badgain[2])))

        badants = badgain
        return badants

    def apply(self, data):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        """

        # flag bad ants
        if self.flagants:
            badants = self.calc_flag()
        else:
            badants = n.array([[]])

        # apply gain correction
        if hasattr(self, 'bandpass'):
            corr = n.ones_like(data)
            flag = n.ones_like(data.real).astype('int')
            chans_uncal = range(len(self.freqs))
            for spwi in range(len(self.spwind)):
                chsize = n.round(self.bpfreq[1]-self.bpfreq[0], 0)
                ww = n.where( (self.freqs >= self.bpfreq[self.spwind[spwi]*1000]) & (self.freqs <= self.bpfreq[(self.spwind[spwi]+1)*1000-1]+chsize) )[0]
                if len(ww) == 0:
                    self.logger.info('Gain solution frequencies not found in data for spw %d.' % (self.spwind[spwi]))
                firstch = ww[0]
                lastch = ww[-1]+1
                for ch in ww:
                    chans_uncal.remove(ch)
                self.logger.info('Combining gain sol from spw=%d with BW chans from %d-%d' % (self.spwind[spwi], firstch, lastch))
                for badant in n.transpose(badants):
                    if badant[1] == spwi:
                        badbl = n.where((badant[0] == n.array(self.ant1ind)) | (badant[0] == n.array(self.ant2ind)))[0]
                        flag[:, badbl, firstch:lastch, badant[2]] = 0

                corr1 = self.gain[self.ant1ind, spwi, :][None, :, None, :] * self.bandpass[self.ant1ind, firstch:lastch, :][None, :, :, :]
                corr2 = (self.gain[self.ant2ind, spwi, :][None, :, None, :] * self.bandpass[self.ant2ind, firstch:lastch, :][None, :, :, :]).conj()

                corr[:, :, firstch:lastch, :] = corr1 * corr2
            if len(chans_uncal):
                self.logger.info('Setting data without bp solution to zero for chans %s.' % (chans_uncal))
                flag[:, :, chans_uncal,:] = 0
            data[:] *= flag/corr
        else:
            for spw in range(len(self.gain[0,0])):
                pass

    def plot(self):
        """ Quick visualization of calibration solution.
        """
        import pylab as p
        p.clf()
        fig = p.figure(1)
        nspw = len(self.gain[0])
        ext = n.ceil(n.sqrt(nspw))  # find best squre plot (simplest)
        for spw in range(len(self.gain[0])):
            ax = fig.add_subplot(ext, ext, spw+1)
            for pol in [0,1]:
                ax.scatter(range(len(self.gain)), n.abs(self.gain.data[:,spw,pol]), color=n.array(['k','y']).take(self.gain.mask[:,spw,pol]), marker=['x','.'][pol])

        fig.show()

def openBpolyFile(caltable, debug=False):
    logger = logging.getLogger(__name__)

#    mytb = au.createCasaTool(tbtool)    # from analysisutilities by corder
    tb.open(caltable)
    desc = tb.getdesc()
    if ('POLY_MODE' in desc):
        polyMode = tb.getcol('POLY_MODE')
        polyType = tb.getcol('POLY_TYPE')
        scaleFactor = tb.getcol('SCALE_FACTOR')
        antenna1 = tb.getcol('ANTENNA1')
        times = tb.getcol('TIME')
        cal_desc_id = tb.getcol('CAL_DESC_ID')
        nRows = len(polyType)
        for pType in polyType:
            if (pType != 'CHEBYSHEV'):
                logger.info("I do not recognized polynomial type = %s" % (pType))
                return
        # Here we assume that all spws have been solved with the same mode
        uniqueTimesBP = n.unique(tb.getcol('TIME'))
        nUniqueTimesBP = len(uniqueTimesBP)
        if (nUniqueTimesBP >= 2):
            logger.debug("Multiple BP sols found with times differing by %s seconds. Using first." % (str(uniqueTimesBP-uniqueTimesBP[0])))
            nUniqueTimesBP = 1
            uniqueTimesBP = uniqueTimesBP[0]
        mystring = ''
        nPolyAmp = tb.getcol('N_POLY_AMP')
        nPolyPhase = tb.getcol('N_POLY_PHASE')
        frequencyLimits = tb.getcol('VALID_DOMAIN')
        increments = 0.001*(frequencyLimits[1,:]-frequencyLimits[0,:])
        frequenciesGHz = []
        for i in range(len(frequencyLimits[0])):
           freqs = (1e-9)*n.arange(frequencyLimits[0,i],frequencyLimits[1,i],increments[i])       # **for some reason this is nch-1 long?**
           frequenciesGHz.append(freqs)
        polynomialAmplitude = []
        polynomialPhase = []
        for i in range(len(polyMode)):
            polynomialAmplitude.append([1])
            polynomialPhase.append([0])
            if (polyMode[i] == 'A&P' or polyMode[i] == 'A'):
                polynomialAmplitude[i]  = tb.getcell('POLY_COEFF_AMP',i)[0][0][0]
            if (polyMode[i] == 'A&P' or polyMode[i] == 'P'):
                polynomialPhase[i] = tb.getcell('POLY_COEFF_PHASE',i)[0][0][0]
  
        tb.close()
        tb.open(caltable+'/CAL_DESC')
        nSpws = len(tb.getcol('NUM_SPW'))
        spws = tb.getcol('SPECTRAL_WINDOW_ID')
        spwBP = []
        for c in cal_desc_id:
            spwBP.append(spws[0][c])
        tb.close()
        nPolarizations = len(polynomialAmplitude[0]) / nPolyAmp[0]
        mystring += '%.3f, ' % (uniqueTimesBP/(24*3600))
        logger.debug('BP solution has unique time(s) %s and %d pols' % (mystring, nPolarizations))
        
        # This value is overridden by the new function doPolarizations in ValueMapping.
        # logger.debug("Inferring %d polarizations from size of polynomial array" % (nPolarizations))
        return([polyMode, polyType, nPolyAmp, nPolyPhase, scaleFactor, nRows, nSpws, nUniqueTimesBP,
                uniqueTimesBP, nPolarizations, frequencyLimits, increments, frequenciesGHz,
                polynomialPhase, polynomialAmplitude, times, antenna1, cal_desc_id, spwBP])
    else:
        tb.close()
        return([])
   # end of openBpolyFile()

def calcChebyshev(coeffs, validDomain, freqs):
    """
    Given a set of coefficients,
    this method evaluates a Chebyshev approximation.
    Used for CASA bandpass reading.
    input coeffs and freqs are numpy arrays
    """

    logger = logging.getLogger(__name__)

    domain = (validDomain[1] - validDomain[0])[0]
    bins = -1 + 2* n.array([ (freqs[i]-validDomain[0,i])/domain for i in range(len(freqs))])
    ncoeffs = len(coeffs[0])/2
    rr = n.array([n.polynomial.chebyshev.chebval(bins[i], coeffs[i,:ncoeffs]) for i in range(len(coeffs))])
    ll = n.array([n.polynomial.chebyshev.chebval(bins[i], coeffs[i,ncoeffs:]) for i in range(len(coeffs))])

    return rr,ll

class telcal_sol():
    """ Instantiated with on telcalfile.
    Parses .GN file and provides tools for applying to data of shape (nints, nbl, nch, npol)
    """

    def __init__(self, telcalfile, flagants=True):
        self.logger = logging.getLogger(__name__)

        if os.path.exists(telcalfile):
            self.parseGN(telcalfile)
            self.logger.info('Read telcalfile %s' % telcalfile)
            if flagants:
                self.flagants()
        else:
            self.logger.warn('Gainfile not found.')
            raise IOError

    def flagants(self, threshold=50):
        """ Flags solutions with amplitude more than threshold larger than median.
        """

        # identify very low gain amps not already flagged
        badsols = n.where( (n.median(self.amp)/self.amp > threshold) & (self.flagged == False))[0]
        if len(badsols):
            self.logger.info('Solutions %s flagged (times %s, ants %s, freqs %s) for low gain amplitude.' % (str(badsols), self.mjd[badsols], self.antname[badsols], self.ifid[badsols]))
            for sol in badsols:
                self.flagged[sol] = True

    def set_selection(self, time, freqs, blarr, calname='', radec=(), dist=0, spwind=[], pols=['XX','YY']):
        """ Set select parameter that defines spectral window, time, or any other selection.
        time (in mjd) defines the time to find solutions near for given calname.
        freqs (in Hz) is frequencies in data.
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        calname defines the name of the calibrator to use. if blank, uses only the time selection.
        pols is from d['pols'] (e.g., ['RR']). single or dual parallel allowed. not yet implemented.
        radec, dist, spwind not used. here for uniformity with casa_sol.
        """

        self.freqs = freqs
        self.chansize = freqs[1]-freqs[0]
        self.select = self.complete   # use only complete solution sets (set during parse)
        self.blarr = blarr
        if spwind:
            self.logger.warn('spwind option not used for telcal_sol. Applied based on freqs.')
        if radec:
            self.logger.warn('radec option not used for telcal_sol. Applied based on calname.')
        if dist:
            self.logger.warn('dist option not used for telcal_sol. Applied based on calname.')

        # define pol index
        if 'X' in ''.join(pols) or 'Y' in ''.join(pols):
            polord = ['XX', 'YY']
        elif 'R' in ''.join(pols) or 'L' in ''.join(pols):
            polord = ['RR', 'LL']
        self.polind = [polord.index(pol) for pol in pols]

        if calname:
            nameselect = []
            for ss in n.unique(self.source[self.select]):
                if calname in ss:
                    nameselect = n.where(self.source[self.select] == ss)   # define selection for name
                    self.select = self.select[nameselect]       # update overall selection
                    self.logger.debug('Selection down to %d solutions with %s' % (len(self.select), calname))
            if not nameselect:
                self.logger.warn('Calibrator name %s not found. Ignoring.' % (calname))

        # select freq
        freqselect = n.where([ff in n.around(self.freqs, -6) for ff in n.around(1e6*self.skyfreq[self.select], -6)])   # takes solution if band center is in (rounded) array of chan freqs
        if len(freqselect[0]) == 0:
            raise StandardError('No complete set of telcal solutions at that frequency.')
        self.select = self.select[freqselect[0]]    # update overall selection
        self.logger.info('Frequency selection cut down to %d solutions' % (len(self.select)))

        # select pol
#        ifids = self.ifid[self.select]
#        if (polstr == 'RR') or (polstr == 'XX'):
#            polselect = n.where(['A' in ifid or 'B' in ifid for ifid in ifids])
#        elif (polstr == 'LL') or (polstr == 'YY'):
#            polselect = n.where(['C' in ifid or 'D' in ifid for ifid in ifids])
#        self.select = self.select[polselect]    # update overall selection
        self.polarization = n.empty(len(self.ifid))
        for i in range(len(self.ifid)):
            if ('A' in self.ifid[i]) or ('B' in self.ifid[i]):
                self.polarization[i] = 0
            elif ('C' in self.ifid[i]) or ('D' in self.ifid[i]):
                self.polarization[i] = 1

        # select by smallest time distance for source
        mjddist = n.abs(time - n.unique(self.mjd[self.select]))
        closest = n.where(mjddist == mjddist.min())
        if len(closest[0]) > 1:
            self.logger.info('Multiple closest solutions in time (%s). Taking first.' % (str(closest[0])))
            closest = closest[0][0]
        timeselect = n.where(self.mjd[self.select] == n.unique(self.mjd[self.select])[closest])   # define selection for time
        self.select = self.select[timeselect[0]]    # update overall selection
        self.logger.info('Selection down to %d solutions separated from given time by %d minutes' % (len(self.select), mjddist[closest]*24*60))

        self.logger.debug('Selected solutions: %s' % str(self.select))
        self.logger.info('MJD: %s' % str(n.unique(self.mjd[self.select])))
        self.logger.debug('Mid frequency (MHz): %s' % str(n.unique(self.skyfreq[self.select])))
        self.logger.debug('IFID: %s' % str(n.unique(self.ifid[self.select])))
        self.logger.info('Source: %s' % str(n.unique(self.source[self.select])))
        self.logger.debug('Ants: %s' % str(n.unique(self.antname[self.select])))

    def parseGN(self, telcalfile, onlycomplete=True):
        """Takes .GN telcal file and places values in numpy arrays.
        onlycomplete defines whether to toss times with less than full set of solutions (one per spw, pol, ant).
        """

        skip = 3   # skip first three header lines
        MJD = 0; UTC = 1; LSTD = 2; LSTS = 3; IFID = 4; SKYFREQ = 5; ANT = 6; AMP = 7; PHASE = 8
        RESIDUAL = 9; DELAY = 10; FLAGGED = 11; ZEROED = 12; HA = 13; AZ = 14; EL = 15
        SOURCE = 16
        #FLAGREASON = 17

        mjd = []; utc = []; lstd = []; lsts = []; ifid = []; skyfreq = []; 
        antname = []; amp = []; phase = []; residual = []; delay = []; 
        flagged = []; zeroed = []; ha = []; az = []; el = []; source = []
        #flagreason = []

        i = 0
        for line in open(telcalfile,'r'):

            fields = line.split()
            if i < skip:
                i += 1
                continue

            if ('NO_ANTSOL_SOLUTIONS_FOUND' in line):
                # keep ERROR solutions now that flagging works
                continue

            try:
                mjd.append(float(fields[MJD])); utc.append(fields[UTC]); lstd.append(float(fields[LSTD])); lsts.append(fields[LSTS])
                ifid.append(fields[IFID]); skyfreq.append(float(fields[SKYFREQ])); antname.append(fields[ANT])
                amp.append(float(fields[AMP])); phase.append(float(fields[PHASE])); residual.append(float(fields[RESIDUAL]))
                delay.append(float(fields[DELAY])); flagged.append('true' == (fields[FLAGGED]))
                zeroed.append('true' == (fields[ZEROED])); ha.append(float(fields[HA])); az.append(float(fields[AZ]))
                el.append(float(fields[EL])); source.append(fields[SOURCE])
#                flagreason.append('')  # 18th field not yet implemented
            except ValueError:
                self.logger.warn('Trouble parsing line of telcal file. Skipping.')
                continue

        self.mjd = n.array(mjd); self.utc = n.array(utc); self.lstd = n.array(lstd); self.lsts = n.array(lsts)
        self.ifid = n.array(ifid); self.skyfreq = n.array(skyfreq); self.antname = n.array(antname); self.amp = n.array(amp) 
        self.phase = n.array(phase); self.residual = n.array(residual); self.delay = n.array(delay)
        self.flagged = n.array(flagged); self.zeroed = n.array(zeroed); self.ha = n.array(ha); self.az = n.array(az)
        self.el = n.array(el); self.source = n.array(source); 
        #self.flagreason = n.array(flagreason)

        # purify list to keep only complete solution sets
        if onlycomplete:
            completecount = len(n.unique(self.ifid)) * len(n.unique(self.antname))
            complete = []
            for mjd in n.unique(self.mjd):
                mjdselect = list(n.where(mjd == self.mjd)[0])
                if len(mjdselect) == completecount:
                    complete = complete + mjdselect
            self.complete = n.array(complete)
        else:
            self.complete = n.arange(len(self.mjd))

        # make another version of ants array
        antnum = []
        for aa in self.antname:
            antnum.append(int(aa[2:]))    # cuts the 'ea' from start of antenna string to get integer
        self.antnum = n.array(antnum)

    def calcgain(self, ant1, ant2, skyfreq, pol):
        """ Calculates the complex gain product (g1*g2) for a pair of antennas.
        """

        select = self.select[n.where( (self.skyfreq[self.select] == skyfreq) & (self.polarization[self.select] == pol) )[0]]

        if len(select):  # for when telcal solutions don't exist
            ind1 = n.where(ant1 == self.antnum[select])
            ind2 = n.where(ant2 == self.antnum[select])
            g1 = self.amp[select][ind1]*n.exp(1j*n.radians(self.phase[select][ind1])) * (not self.flagged.astype(int)[select][ind1][0])
            g2 = self.amp[select][ind2]*n.exp(-1j*n.radians(self.phase[select][ind2])) * (not self.flagged.astype(int)[select][ind2][0])
        else:
            g1 = [0]; g2 = [0]

        try:
            assert (g1[0] != 0j) and (g2[0] != 0j)
            invg1g2 = 1./(g1[0]*g2[0])
        except (AssertionError, IndexError):
            invg1g2 = 0
        return invg1g2

    def calcdelay(self, ant1, ant2, skyfreq, pol):
        """ Calculates the relative delay (d1-d2) for a pair of antennas in ns.
        """

        select = self.select[n.where( (self.skyfreq[self.select] == skyfreq) & (self.polarization[self.select] == pol) )[0]]

        ind1 = n.where(ant1 == self.antnum[select])
        ind2 = n.where(ant2 == self.antnum[select])
        d1 = self.delay[select][ind1]
        d2 = self.delay[select][ind2]
        if len(d1-d2) > 0:
            return d1-d2
        else:
            return n.array([0])

    def apply(self, data):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        """

        # find best skyfreq for each channel
        skyfreqs = n.unique(self.skyfreq[self.select])    # one per spw
        nch_tot = len(self.freqs)
        chan_bandnum = [range(nch_tot*i/len(skyfreqs), nch_tot*(i+1)/len(skyfreqs)) for i in range(len(skyfreqs))]  # divide chans by number of spw in solution
        self.logger.info('Solutions for %d spw: (%s)' % (len(skyfreqs), skyfreqs))

        for j in range(len(skyfreqs)):
            skyfreq = skyfreqs[j]
            chans = chan_bandnum[j]
            self.logger.info('Applying gain solution for chans from %d-%d' % (chans[0], chans[-1]))

            # define freq structure to apply delay solution
            nch = len(chans)
            chanref = nch/2    # reference channel at center
            relfreq = self.chansize*(n.arange(nch) - chanref)   # relative frequency

            for i in range(len(self.blarr)):
                ant1, ant2 = self.blarr[i]  # ant numbers (1-based)
                for pol in self.polind:
                    # apply gain correction
                    invg1g2 = self.calcgain(ant1, ant2, skyfreq, pol)
                    data[:,i,chans,pol-self.polind[0]] = data[:,i,chans,pol-self.polind[0]] * invg1g2    # hack: lousy data pol indexing

                    # apply delay correction
                    d1d2 = self.calcdelay(ant1, ant2, skyfreq, pol)
                    delayrot = 2*n.pi*(d1d2[0] * 1e-9) * relfreq      # phase to rotate across band
                    data[:,i,chans,pol-self.polind[0]] = data[:,i,chans,pol-self.polind[0]] * n.exp(-1j*delayrot[None, None, :])     # do rotation
