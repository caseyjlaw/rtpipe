import numpy as n
import os, glob, sys
import casautil
import logging, logging.config

# set up
tb = casautil.tools.table()

LOGGING = {
    'version': 1,              
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level':'INFO',    
            'class':'logging.StreamHandler',
        },  
        'file': {
            'level':'INFO',    
            'class':'logging.FileHandler',
            'filename':'tmp.log',
        },  
    },
    'loggers': {
        '': {                  
            'handlers': ['default', 'file'],        
            'level': 'INFO',  
            'propagate': True  
        }
    }
}

class casa_sol():
    """ Container for CASA caltable(s).
    Provides tools for applying to data of shape (nints, nbl, nch, npol).
    Initialize class based on input file(s) and selection criteria.
    Optional flux scale gain file can be given. Should be gain file applied to source with setjy applied.
    """

    def __init__(self, gainfile, flagants=True):
        """ Initialize with a table of CASA gain solutions. Can later add BP.
        """
        
        if os.path.exists(gainfile):
            self.parsegain(gainfile)
            self.flagants = flagants

            # custom log file
            logger = logging.getLogger('cal')
            LOGGING['handlers']['file']['filename'] = 'calibration_' + os.path.split(gainfile)[1].split('.g')[0] + '.txt'
            logging.config.dictConfig(LOGGING)
        else:
            logging.warning('Gainfile not found.')
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
        #     print 'Found parent data MS %s' % msfile
        # except IndexError:
        #     print 'Could not find parent data MS for metadata...'

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

        logging.info('Parsed gain table solutions for %d solutions (skipping %d), %d ants, %d spw, and %d pols' % (nsol, len(skip), nants, nspw, npol))
        logging.info('Unique solution times: %s' % str(self.uniquemjd))

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
        logging.info('Parsed bp solutions for %d solutions, %d ants, %d spw, and %d pols' % (nUniqueTimesBP, nants, nSpws, nPolarizations))
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
#        print 'Parsed bp table solutions for %d solutions, %d ants, %d spw, and %d pols' % (nUniqueTimesBP, nants, nSpws, nPolarizations)

    def setselection(self, time, freqs, radec=(), dist=10., spws=[0,1], pols=[0,1], verbose=0):
        """ Set select parameter that defines time, spw, and pol solutions to apply.
        time defines the time to find solutions near in mjd.
        freqs defines frequencies to select bandpass solution
        radec (radian tuple) and dist (deg) define optional location of source for filtering solutions.
        spws is list of min/max indices to be used (e.g., [0,1])
        pols is index of polarizations.
        pols/spws not yet implemented beyond 2sb, 2pol.
        """

        # spw and pols selection not yet implemented beyond 2/2
        self.spws = spws
        self.pols = pols

        # select by smallest time distance for source within some angular region of target
        if len(radec):
            ra, dec = radec
            calra = n.array([self.radec[i][0] for i in range(len(self.radec))])
            caldec = n.array([self.radec[i][1] for i in range(len(self.radec))])
            fields = n.where( (calra - ra < n.radians(dist)) & (caldec - dec < n.radians(dist)) )[0]
            if len(fields) == 0:
                logging.info('Warning: no close calibrator found. Removing radec restriction.')
                fields = n.unique(self.uniquefield)
        else:
            fields = n.unique(self.uniquefield)

        sel = []
        for field in fields:
            sel += list(n.where(field == self.uniquefield)[0])
        mjddist = n.abs(time - self.uniquemjd[sel])
        closestgain = n.where(mjddist == mjddist.min())[0][0]

        logging.info('Using gain solution for field %d at MJD %.5f, separated by %d min ' % (self.uniquefield[n.where(self.uniquemjd == self.uniquemjd[sel][closestgain])], self.uniquemjd[closestgain], mjddist[closestgain]*24*60))
        self.gain = self.gain[closestgain,:,spws[0]:spws[1]+1,pols[0]:pols[1]+1]

        if hasattr(self, 'bandpass'):
            bins = [n.where(n.min(n.abs(self.bpfreq-selfreq)) == n.abs(self.bpfreq-selfreq))[0][0] for selfreq in freqs]
            self.bandpass = self.bandpass[:,bins,pols[0]:pols[1]+1]
            self.freqs = freqs
            logging.debug('Using solution at BP bins: ', bins)

    def calc_flag(self, sig=3.0):
        """ Calculates antennas to flag, based on bad gain and bp solutions.
        """
 
        if len(self.gain.shape) == 4:
            gamp = n.abs(self.gain).mean(axis=0)   # mean gain amp for each ant over time
        elif len(self.gain.shape) == 3:
            gamp = n.abs(self.gain)   # gain amp for selected time

#        badgain = n.where(gamp < gamp.mean() - sig*gamp.std())
        badgain = n.where( (gamp < n.median(gamp) - sig*gamp.std()) | gamp.mask)
        logging.info('Flagging low/bad gains for ant/spw/pol: %s %s %s' % (str(self.antnum[badgain[0]]), str(badgain[1]), str(badgain[2])))

        badants = badgain
        return badants

    def apply(self, data, blarr):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        """

        ant1ind = [n.where(ant1 == n.unique(blarr))[0][0] for (ant1,ant2) in blarr]
        ant2ind = [n.where(ant2 == n.unique(blarr))[0][0] for (ant1,ant2) in blarr]

        # flag bad ants
        if self.flagants:
            badants = self.calc_flag()
        else:
            badants = n.array([[]])

        # apply gain correction
        if hasattr(self, 'bandpass'):
            corr = n.ones_like(data)
            flag = n.ones_like(data).astype('int')
            chans_uncal = range(len(self.freqs))
            for spw in range(len(self.gain[0])):
                chsize = n.round(self.bpfreq[1]-self.bpfreq[0], 0)
                ww = n.where( (self.freqs >= self.bpfreq[spw*1000]) & (self.freqs <= self.bpfreq[(spw+1)*1000-1]+chsize) )[0]
                if len(ww) == 0:
                    logging.info('Gain solution frequencies not found in data for spw %d.' % (self.spws[spw]))
                firstch = ww[0]
                lastch = ww[-1]+1
                for ch in ww:
                    chans_uncal.remove(ch)
                logging.info('Combining gain sol from spw=%d with BW chans from %d-%d' % (self.spws[spw], firstch, lastch))
                for badant in n.transpose(badants):
                    if badant[1] == spw:
                        badbl = n.where((badant[0] == n.array(ant1ind)) | (badant[0] == n.array(ant2ind)))[0]
                        flag[:, badbl, firstch:lastch, badant[2]] = 0

                corr1 = self.gain[ant1ind, spw, :][None, :, None, :] * self.bandpass[ant1ind, firstch:lastch, :][None, :, :, :]
                corr2 = (self.gain[ant2ind, spw, :][None, :, None, :] * self.bandpass[ant2ind, firstch:lastch, :][None, :, :, :]).conj()

                corr[:, :, firstch:lastch, :] = corr1 * corr2
            if len(chans_uncal):
                logging.info('Setting data without bp solution to zero for chans %s.' % (chans_uncal))
                flag[:, :, chans_uncal,:] = 0
            data[:] *= flag/corr
        else:
            for spw in range(len(self.gain[0,0])):
                pass

def openBpolyFile(caltable, debug=False):
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
                logging.info("I do not recognized polynomial type = %s" % (pType))
                return
        # Here we assume that all spws have been solved with the same mode
        uniqueTimesBP = n.unique(tb.getcol('TIME'))
        nUniqueTimesBP = len(uniqueTimesBP)
        if (nUniqueTimesBP >= 2):
            logging.debug("Multiple BP sols found with times differing by %s seconds. Using first." % (str(uniqueTimesBP-uniqueTimesBP[0])))
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
        logging.debug('BP solution has unique time(s) %s and %d pols' % (mystring, nPolarizations))
        
        # This value is overridden by the new function doPolarizations in ValueMapping.
        # print "Inferring %d polarizations from size of polynomial array" % (nPolarizations)
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

    domain = (validDomain[1] - validDomain[0])[0]
    bins = -1 + 2* n.array([ (freqs[i]-validDomain[0,i])/domain for i in range(len(freqs))])
    ncoeffs = len(coeffs[0])/2
    rr = n.array([n.polynomial.chebyshev.chebval(bins[i], coeffs[i,:ncoeffs]) for i in range(len(coeffs))])
    ll = n.array([n.polynomial.chebyshev.chebval(bins[i], coeffs[i,ncoeffs:]) for i in range(len(coeffs))])

    return rr,ll

class telcal_sol():
    """ Container for telcal solutions. Parses .GN files and provides tools for applying to data of shape (nints, nbl, nch, npol)
    Initialize class based on telcalfile and selection criteria.
    solnum is iteration of solution (0-based), pol is 0/1 for R/L.
    freqs is array of channel frequencies in Hz. Should be something like tpipe.freq*1e9.
    """

    def __init__(self, telcalfile, freqs=[1.4e9,1.401e9]):
        self.freqs = freqs
        self.chansize = freqs[1]-freqs[0]
        self.parseGN(telcalfile)
        logging.info('Read telcalfile %s' % telcalfile)

    def setselection(self, calname, time, polstr, verbose=0):
        """ Set select parameter that defines spectral window, time, or any other selection.
        calname defines the name of the calibrator to use. if blank, uses only the time selection.
        time defines the time to find solutions near for given calname. it is in mjd.
        polstr is either 'RR' or 'LL', where (A,C) == (R,L), it seems.
        """

        self.select = self.complete   # use only complete solution sets (set during parse)

        if calname:
            nameselect = []
            for ss in n.unique(self.source[self.select]):
                if calname in ss:
                    nameselect = n.where(self.source[self.select] == ss)   # define selection for name
                    self.select = self.select[nameselect[0]]       # update overall selection
                    if verbose:
                        logging.info('Selection down to %d solutions with %s' % (len(self.select), calname))
            if len(nameselect) == 0:
                logging.info('Calibrator name %s not found. Ignoring.' % (calname))

        # select freq
        freqselect = n.where( n.around(1e6*self.skyfreq[self.select],-6) == n.around(self.freqs[len(self.freqs)/2],-6) )   # define selection for time
        if len(freqselect[0]) == 0:
            raise StandardError('No complete set of telcal solutions at that frequency.')
        self.select = self.select[freqselect[0]]    # update overall selection
        if verbose:
            logging.info('Frequency selection cut down to %d solutions' % (len(self.select)))

        # select pol
        ifids = self.ifid[self.select]
        for pp in n.unique(ifids):
            if (('A' in pp or 'B' in pp) and ((polstr == 'RR') or (polstr == 'XX'))):
                polselect = n.where(ifids == pp)
            elif (('C' in pp or 'D' in pp) and ((polstr == 'LL') or (polstr == 'YY'))):
                polselect = n.where(ifids == pp)

        self.select = self.select[polselect[0]]    # update overall selection

        # select by smallest time distance for source
        mjddist = n.abs(time - n.unique(self.mjd[self.select]))
        closest = n.where(mjddist == mjddist.min())
        timeselect = n.where(self.mjd[self.select] == n.unique(self.mjd[self.select])[closest])   # define selection for time
        self.select = self.select[timeselect[0]]    # update overall selection
        logging.debug('Selection down to %d solutions separated from given time by %d minutes' % (len(self.select), mjddist[closest]*24*60))

        if verbose:
            logging.info('Selected solutions: %s' % str(self.select))
            logging.info('MJD: %s' % str(n.unique(self.mjd[self.select])))
            logging.info('Mid frequency (MHz): %s' % str(n.unique(self.skyfreq[self.select])))
            logging.info('IFID: %s' % str(n.unique(self.ifid[self.select])))
            logging.info('Source: %s' % str(n.unique(self.source[self.select])))
            logging.info('Ants: %s' % str(n.unique(self.antname[self.select])))

    def parseGN(self, telcalfile):
        """Takes .GN telcal file and places values in numpy arrays.
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

            if ('NO_ANTSOL_SOLUTIONS_FOUND' in line) or ('ERROR' in line):
                continue

            mjd.append(float(fields[MJD])); utc.append(fields[UTC]); lstd.append(float(fields[LSTD])); lsts.append(fields[LSTS])
            ifid.append(fields[IFID]); skyfreq.append(float(fields[SKYFREQ])); antname.append(fields[ANT])
            amp.append(float(fields[AMP])); phase.append(float(fields[PHASE])); residual.append(float(fields[RESIDUAL]))
            delay.append(float(fields[DELAY])); flagged.append('true' == (fields[FLAGGED]))
            zeroed.append('true' == (fields[ZEROED])); ha.append(float(fields[HA])); az.append(float(fields[AZ]))
            el.append(float(fields[EL])); source.append(fields[SOURCE])
#            flagreason.append('')  # 18th field not yet implemented

        self.mjd = n.array(mjd); self.utc = n.array(utc); self.lstd = n.array(lstd); self.lsts = n.array(lsts)
        self.ifid = n.array(ifid); self.skyfreq = n.array(skyfreq); self.antname = n.array(antname); self.amp = n.array(amp) 
        self.phase = n.array(phase); self.residual = n.array(residual); self.delay = n.array(delay)
        self.flagged = n.array(flagged); self.zeroed = n.array(zeroed); self.ha = n.array(ha); self.az = n.array(az)
        self.el = n.array(el); self.source = n.array(source); 
        #self.flagreason = n.array(flagreason)

        # purify list to keep only complete solution sets
#        uu = n.unique(self.mjd)
#        uu2 = n.concatenate( (uu, [uu[-1] + (uu[-1]-uu[-2])]) )  # add rightmost bin
#        count,bin = n.histogram(self.mjd, bins=uu2)
#        goodmjd = bin[n.where(count == count.max())]
#        complete = n.array([], dtype='int')
#        for mjd in goodmjd:
#            complete = n.concatenate( (complete, n.where(mjd == self.mjd)[0]) )
#        self.complete = n.array(complete)
        self.complete = n.arange(len(self.mjd))

        # make another version of ants array
        antnum = []
        for aa in self.antname:
            antnum.append(int(aa[2:]))    # cuts the 'ea' from start of antenna string to get integer
        self.antnum = n.array(antnum)

    def calcgain(self, ant1, ant2):
        """ Calculates the complex gain product (g1*g2) for a pair of antennas.
        """

        ind1 = n.where(ant1 == self.antnum[self.select])
        ind2 = n.where(ant2 == self.antnum[self.select])
        g1 = self.amp[self.select][ind1]*n.exp(1j*n.radians(self.phase[self.select][ind1]))
        g2 = self.amp[self.select][ind2]*n.exp(-1j*n.radians(self.phase[self.select][ind2]))
        if len(g1*g2) > 0:
            invg1g2 = 1/(g1*g2)
            invg1g2[n.where( (g1 == 0j) | (g2 == 0j) )] = 0.
            return invg1g2
        else:
            return n.array([0])

    def calcdelay(self, ant1, ant2):
        """ Calculates the relative delay (d1-d2) for a pair of antennas in ns.
        """

        ind1 = n.where(ant1 == self.antnum[self.select])
        ind2 = n.where(ant2 == self.antnum[self.select])
        d1 = self.delay[self.select][ind1]
        d2 = self.delay[self.select][ind2]
        if len(d1-d2) > 0:
            return d1-d2
        else:
            return n.array([0])

    def apply(self, data, blarr, pol):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        pol is an index to apply solution (0/1)
        """

        # define freq structure to apply delay solution
        nch = data.shape[2]
        chanref = nch/2    # reference channel at center
        freqarr = self.chansize*(n.arange(nch) - chanref)   # relative frequency

        for i in range(len(blarr)):
            ant1, ant2 = blarr[i]  # ant numbers (1-based)

            # apply gain correction
            invg1g2 = self.calcgain(ant1, ant2)
            data[:,i,:,pol] = data[:,i,:,pol] * invg1g2[0]

            # apply delay correction
            d1d2 = self.calcdelay(ant1, ant2)
            delayrot = 2*n.pi*(d1d2 * 1e-9)*freqarr      # phase to rotate across band
            data[:,i,:,pol] = data[:,i,:,pol] * n.exp(-1j*delayrot[None, :])     # do rotation
