#
# Define object for parsing parameters for the rtpipe pipeline
# claw, 16jun15
#

class Params(object):
    """ Parameter class used as input to define pipeline
    Lets system use python-like file for importing into namespace and same file for oo-like parameter definition.
    Uses 'exec', for which I am not proud.
    Also defines default values for all parameters.
    """

    def  __init__(self, paramfile=''):
    
        # default values
        self.chans = []; self.spw = []    
        self.nskip = 0; self.excludeants = []; self.read_tdownsample = 1; self.read_fdownsample = 1
        self.selectpol = ['RR', 'LL', 'XX', 'YY']   # default processing assumes dual-pol
        self.nthread = 1; self.nchunk = 0; self.nsegments = 0; self.scale_nsegments = 1
        self.timesub = ''
        self.dmarr = []; self.dtarr = [1]    # dmarr = [] will autodetect, given other parameters
        self.dm_maxloss = 0.05; self.maxdm = 0; self.dm_pulsewidth = 3000   # dmloss is fractional sensitivity loss, maxdm in pc/cm3, width in microsec
        self.searchtype = 'image1'; self.sigma_image1 = 7.; self.sigma_image2 = 7.
        self.l0 = 0.; self.m0 = 0.
        self.uvres = 0; self.npix = 0; self.npix_max = 0; self.uvoversample = 1.
        self.flaglist = [('badchtslide', 4., 0.) , ('badap', 3., 0.2), ('blstd', 3.0, 0.05)]
        self.flagantsol = True; self.gainfile = ''; self.bpfile = ''; self.fileroot = ''; self.applyonlineflags = True
        self.savenoise = False; self.savecands = False; self.logfile = True; self.loglevel = 'INFO'
        self.writebdfpkl = False; self.mock = 0
                           
        # overload with the parameter file values, if provided
        if len(paramfile):
            self.parse(paramfile)


    def parse(self, paramfile):
        """ Read parameter file and set parameter values.
        File should have python-like syntax. Full file name needed.
        """

        with open(paramfile, 'r') as f:
            for line in f.readlines():
                line_clean = line.rstrip('\n').split('#')[0]   # trim out comments and trailing cr
                if line_clean and '=' in line:   # use valid lines only
                    attribute, value = line_clean.split('=')
                    try:
                        value_eval = eval(value.strip())
                    except NameError:
                        value_eval = value.strip()
                    finally:
                        setattr(self, attribute.strip(), value_eval)


    @property
    def defined(self):
        return self.__dict__.keys()


    def __getitem__(self, key):
        return self.__dict__[key]


    def __str__(self):
        return str(self.__dict__)


    def __repr__(self):
        return str(self.__dict__)
