from ipywidgets import interact, FloatSlider, Text, Dropdown, Output, VBox, fixed
import pickle, os
from IPython.display import display, Javascript

class state:
    """ Jupyter notebook state attached to notebook name.
    Useful when running programatically and interactively and need to save parameters for next user.
    """

    def __init__(self, statedir):
        """ Initialize with directory to save state as files """

        self.statedir = statedir

        if not os.path.exists(self.statedir): os.mkdir(self.statedir)


    @property
    def objects(self):
        """ List names of stored objects """
        return os.listdir(self.statedir)


    def save(self, obj, label, format='text'):
        """ Save or update obj as pkl file with name label 

        format can be 'text' or 'pickle'.
        """

        # initialize hidden state directory

        objloc = '{0}/{1}'.format(self.statedir, label)

        with open(objloc, 'w') as fp:
            if format == 'pickle':
                pickle.dump(obj, fp)
            elif format == 'text':
                fp.write(str(obj))
            else:
                print('Format {0} not recognized. Please choose either pickle or text.'.format(format))

        print('Saving {0} to label {1}'.format(obj, label))


    def load(self, label):
        """ Load obj with give label from hidden state directory """

        objloc = '{0}/{1}'.format(self.statedir, label)

        try:
            obj = pickle.load(open(objloc, 'r')) 
        except (KeyError, IndexError, EOFError):
            obj = open(objloc, 'r').read()
            try:
                obj = float(obj)
            except ValueError:
                pass
        except IOError:
            obj = None

        return obj


    def setText(self, label, default='', description='Set Text', format='text'):
        """ Set text in a notebook pipeline (via interaction or with nbconvert) """

        obj = self.load(label)
        if obj == None:
            obj=default
            self.save(obj, label)  # initialize with default

        textw = Text(value=obj, description=description)
        hndl = interact(self.save, obj=textw, label=fixed(label), format=fixed(format))


    def setFloat(self, label, default=0, min=-20, max=20, description='Set Float', format='text'):
        """ Set float in a notebook pipeline (via interaction or with nbconvert) """

        obj = self.load(label)
        if obj == None:
            obj=default
            self.save(obj, label)  # initialize with default

        floatw = FloatSlider(value=obj, min=min, max=max, description=description)
        hndl = interact(self.save, obj=floatw, label=fixed(label), format=fixed(format))


    def setDropdown(self, label, default=None, options=[], description='Set Dropdown', format='text'):
        """ Set float in a notebook pipeline (via interaction or with nbconvert) """

        obj = self.load(label)
        if obj == None:
            obj=default
            self.save(obj, label)  # initialize with default

        dropdownw = Dropdown(value=obj, options=options, description=description)
        hndl = interact(self.save, obj=dropdownw, label=fixed(label), format=fixed(format))


def getnbname():
    """ Hack to get name of notebook as python obj 'nbname'. Does not work with 'run all' """

    display(Javascript("""IPython.notebook.kernel.execute("nbname = " + "\'"+IPython.notebook.notebook_name+"\'");"""))    
