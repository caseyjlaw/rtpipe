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
        self.dmarr = [0]; self.dtarr = [1]
        self.nskip = 0; self.excludeants = []; self.read_downsample = 1
        self.nthread = 8; self.nchunk = 0
        self.timesub = ''
        self.searchtype = 'image1'; self.sigma_image1 = 7.; self.sigma_image2 = 7.
        self.l0 = 0.; self.m0 = 0.
        self.uvres = 0; self.npix = 0
        self.flagmode = ''; self.flagantsol = True
        self.gainfile = ''; self.bpfile = ''
        self.noisefile = ''; self.candsfile = ''
                           
        # overload with the parameter file values, if provided
        if len(paramfile):
            self.parse(paramfile)

    def parse(self, paramfile):
        """ Read parameter file and set parameter values.
        File should have python-like syntax. Full file name needed.
        """

        f = open(paramfile, 'r')
        for line in f.readlines():
            line = line.rstrip('\n').split('#')[0]   # trim out comments and trailing cr
            if '=' in line:   # use valid lines only
                exec(line)
                exec('self.'+line.lstrip())

    @property
    def defined(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
