from ipywidgets import interact, FloatSlider, Text, Dropdown, Button, Output, VBox, fixed
import pickle, os
from IPython.display import display, Javascript


def save(obj, label, format='text', statedir='.nbpipeline'):
    """ Save or update obj as pkl file with name label 

    format can be 'text' or 'pickle'.
    statedir is the directory name for saving object
    """

    # initialize hidden state directory
    if not os.path.exists(statedir): os.mkdir(statedir)

    objloc = '{0}/{1}'.format(statedir, label)

    with open(objloc, 'w') as fp:
        if format == 'pickle':
            pickle.dump(obj, fp)
        elif format == 'text':
            fp.write(str(obj))


def read(label, statedir='.nbpipeline'):
    """ Read obj with give label from hidden state directory """

    objloc = '{0}/{1}'.format(statedir, label)

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


def list(statedir = '.nbpipeline'):
    """ List names of stored objects """

    print(os.listdir(statedir))


def setText(label, default='', description='Set Text', format='text', statedir='.nbpipeline'):
    """ Set text in a notebook pipeline (via interaction or with nbconvert) """

    obj = read(label)
    if obj == None:
        obj=default
        save(obj, label)  # initialize with default

    textw = Text(value=obj, description=description)
    hndl = interact(save, obj=textw, label=fixed(label), format=fixed(format), statedir=fixed(statedir))


def setFloat(label, default=0, min=-20, max=20, description='Set Float', format='text', statedir='.nbpipeline'):
    """ Set float in a notebook pipeline (via interaction or with nbconvert) """

    obj = read(label)
    if obj == None:
        obj=default
        save(obj, label)  # initialize with default

    floatw = FloatSlider(value=obj, min=min, max=max, description=description)
    hndl = interact(save, obj=floatw, label=fixed(label), format=fixed(format), statedir=fixed(statedir))


def setDropdown(label, default=None, options=[], description='Set Dropdown', format='text', statedir='.nbpipeline'):
    """ Set float in a notebook pipeline (via interaction or with nbconvert) """

    obj = read(label)
    if obj == None:
        obj=default
        save(obj, label)  # initialize with default

    dropdownw = Dropdown(value=obj, options=options, description=description)
    hndl = interact(save, obj=dropdownw, label=fixed(label), format=fixed(format), statedir=fixed(statedir))


def setButton(description=''):
    """ Not working... Not clear how onclick function is called... """

    button=Button(description=description)
    out=Output()
    vbox=VBox(children=(button, out))
    display(vbox)

    def onclick(b):
        interactive.filterdata(data=data, plinds=plinds, d=d, threshold=nbpipeline.read('threshold'),
                           ignorestr=nbpipeline.read('ignorestr'), thresh0=6., thresh1=7.)

    button.on_click(onclick)


def getnbname():
    """ Hack to get name of notebook as python obj 'nbname'. Does not work with 'run all' """

    display(Javascript("""IPython.notebook.kernel.execute("nbname = " + "\'"+IPython.notebook.notebook_name+"\'");"""))    
