import tasklib as tl
import rtpipe.parsesdm as ps
import os, string, glob
from sdmreader import sdmreader

class pipe():
    def __init__(self, sdmfile):
        self.sdmfile = sdmfile
        self.workdir = string.join(sdmfile.rstrip('/').split('/')[:-1], '/') + '/'
        if self.workdir == '/':
            self.workdir = os.getcwd() + '/'

        self.scans, self.sources = sdmreader.read_metadata(sdmfile)
        gainscans = [sc for sc in self.scans.keys() if 'PHASE' in self.scans[sc]['intent']]   # get all cal fields
        bpscans = [sc for sc in self.scans.keys() if 'BANDPASS' in self.scans[sc]['intent']]   # get all cal fields
        self.gainscans = gainscans
        self.bpscans = bpscans
        self.gainstr = string.join([str(ss) for ss in gainscans], ',')
        self.bpstr = string.join([str(ss) for ss in bpscans], ',')
        self.allstr = string.join([str(ss) for ss in sorted(list(set(gainscans+bpscans)))], ',')
        
        if len(self.gainstr) or len(self.bpstr):
            print 'Found gaincal scans %s and bpcal scans %s.' % (self.gainstr, self.bpstr)

    def genms(self, fileroot, scans=[]):
        """ Generate an MS that contains all calibrator scans with 1 s integration time.
        """

        if len(scans):
            scanstr = string.join([str(ss) for ss in sorted(scans)], ',')
        else:
            scanstr = self.allstr

        print 'Splitting out all cal scans (%s) with 1s int time' % scanstr
        newname = ps.sdm2ms(self.sdmfile, fileroot, scanstr, inttime='1')   # integrate down to 1s during split

        return newname

    def run(self, fileroot, gainscans=[], bpscans=[], refant=['ea10'], antsel=[], uvrange='', fluxname0='', fluxname='', spw0='', spw1='', flags=[]):
        """ Run calibration pipeline
        fileroot is root of calibration file names.
        gainscans and bpscans are lists of scan numbers to use (as ints)
        refant is list of antenna name strings (e.g., ['ea10']).
        antsel is list of ants to use (or reject) (e.g., ['!ea08'])
        uvrange is string giving uvrange (e.g., '<5klambda')
        fluxname0 is name of flux calibrator (e.g., '3C48')
        fluxname is name used by CASA (e.g., '0137+331=3C48')
        spw0 is spw selection for gain cal before bp cal (e.g., '0~1:60~75')
        spw1 is spw selection for gain cal after bp cal (e.g., '0~1:6~122')
        flags is the list of flag commands (e.g., ["mode='unflag'", "mode='shadow'", "mode='manual' antenna='ea11'"])
        """

        antposname = fileroot + '.antpos'   # antpos
        delayname = fileroot + '.delay'   # delay cal
        g0name = fileroot + '.g0'   # initial gain correction before bp
        b1name = fileroot + '.b1'   # bandpass file
        g1name = fileroot + '.g1'   # gain cal per scan
        g2name = fileroot + '.g2'   # flux scale applied

        os.chdir(self.workdir)

        # set up scans to use
        if gainscans:
            gainstr = string.join([str(ss) for ss in gainscans], ',')
        else:
            gainstr = self.gainstr
        if bpscans:
            bpstr = string.join([str(ss) for ss in bpscans], ',')
        else:
            bpstr = self.bpstr
        if gainscans or bpscans:
            allstr = string.join([str(ss) for ss in sorted(list(set(gainscans+bpscans)))], ',')
        else:
            allstr = self.allstr

        if fluxname0:
            fluxmodel = '/home/casa/packages/RHEL5/release/casapy-41.0.24668-001-64b/data/nrao/VLA/CalModels/'+fluxname0+'_L.im'
        else:
            fluxmodel = ''

        # set up MS file
        msfile = self.genms(fileroot)

        # flag data via text file
        flfile = open('flags.txt','w')
        for flag in flags:
            flfile.write(flag + '\n')
        flfile.close()

        print 'Flagging with these commands:'
        for ff in enumerate(open('flags.txt')): print ff[1].rstrip()

        cfg = tl.FlaglistConfig()  # configure split
        cfg.vis = msfile
        cfg.inpfile = "flags.txt"
        tl.flaglist(cfg)  # run task

        # clean up
        os.remove('flags.txt')

        # Calibrate!
        if fluxmodel:
            if not os.path.exists(g0name):
                print 'Applying flux model for BP calibrator...'
                cfg = tl.SetjyConfig()
                cfg.vis = msfile
                cfg.scan = bpstr
                cfg.modimage = fluxmodel
                cfg.standard = 'Perley-Butler 2010'    # for some reason 2013 standard can't find 3C48
                tl.setjy(cfg)

                print 'Starting initial gain cal...'
                cfg = tl.GaincalConfig()
                cfg.vis = msfile
                cfg.caltable = g0name
                cfg.gaintable = []
                cfg.scan = bpstr
                cfg.gaintype = 'G'
                cfg.solint = 'inf'
                cfg.spw = spw0
                cfg.refant = refant
                cfg.minsnr = 5.
                cfg.calmode = 'p'
                cfg.antenna = antsel
                cfg.uvrange = uvrange
                tl.gaincal(cfg)

            if not os.path.exists(b1name):
                print 'Starting bp cal...'
                cfg = tl.GaincalConfig()
                cfg.vis = msfile
                cfg.caltable = b1name
                cfg.gaintable = [g0name]
                cfg.scan = bpstr
                cfg.spw = spw1
                cfg.gaintype = 'BPOLY'
                cfg.degamp = 5
                cfg.degphase = 2
                cfg.maskedge = 6
                cfg.solint = 'inf'
                cfg.combine = ['scan']
                cfg.solnorm = True
                cfg.refant = refant
                cfg.antenna = antsel
                cfg.uvrange = uvrange
                tl.gaincal(cfg)

            if not os.path.exists(g1name) or not os.path.exists(g2name):
                print 'Starting gain cal...'
                cfg = tl.GaincalConfig()
                cfg.vis = msfile
                cfg.caltable = g1name
                cfg.gaintable = [b1name]
                cfg.scan = allstr
                cfg.gaintype = 'G'
                cfg.solint = 'inf'
                cfg.spw = spw1
                cfg.refant = refant
                cfg.minsnr = 5.
                cfg.calmode='ap'
                cfg.antenna = antsel
                cfg.uvrange = uvrange
                tl.gaincal(cfg)

                print 'Transferring flux scale...'
                cfg = tl.FluxscaleConfig()
                cfg.vis = msfile
                cfg.caltable = g1name
                cfg.fluxtable = g2name
                cfg.reference = fluxname
                tl.fluxscale(cfg)

        else:    # without fluxscale
            if not os.path.exists(g0name):
                print 'Starting initial gain cal...'
                cfg = tl.GaincalConfig()
                cfg.vis = msfile
                cfg.caltable = g0name
                cfg.gaintable = []
                cfg.scan = bpstr
                cfg.gaintype = 'G'
                cfg.solint = 'inf'
                cfg.spw = spw0
                cfg.refant = refant
                cfg.minsnr = 5.
                cfg.calmode = 'p'
                cfg.antenna = antsel
                cfg.uvrange = uvrange
                tl.gaincal(cfg)

            if not os.path.exists(b1name):
                print 'Starting bp cal...'
                cfg = tl.GaincalConfig()
                cfg.vis = msfile
                cfg.caltable = b1name
                cfg.gaintable = [g0name]
                cfg.scan = bpstr
                cfg.spw = spw1
                cfg.gaintype = 'BPOLY'
                cfg.degamp = 5
                cfg.degphase = 2
                cfg.maskedge = 6
                cfg.solint = 'inf'
                cfg.combine = ['scan']
                cfg.solnorm = True
                cfg.refant = refant
                cfg.antenna = antsel
                cfg.uvrange = uvrange
                tl.gaincal(cfg)

            if not os.path.exists(g1name):
                print 'Starting gain cal...'
                cfg = tl.GaincalConfig()
                cfg.vis = msfile
                cfg.caltable = g1name
                cfg.gaintable = [b1name]
                cfg.scan = allstr
                cfg.gaintype = 'G'
                cfg.solint = 'inf'
                cfg.spw = spw1
                cfg.refant = refant
                cfg.minsnr = 5.
                cfg.calmode='ap'
                cfg.antenna = antsel
                cfg.uvrange = uvrange
                tl.gaincal(cfg)

        return 0
