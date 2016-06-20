__all__ = ['RT', 'parsesdm', 'parsems', 'parsecands', 'parsecal', 'parseparams', 'interactive', 'nbpipeline', 'reproduce']

from rtpipe import *
import os.path

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_notebook(path):
    return os.path.join(_ROOT, 'notebooks', path)
