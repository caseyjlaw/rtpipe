from setuptools import setup, find_packages
setup(
    name = 'rtpipe',
    description = 'Python scripts for fast transient searches with radio interferometer data',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '1.0',
    packages = find_packages(),        # get all python scripts in realtime
    dependency_links = ['http://github.com/caseyjlaw/sdmpy', 'http://github.com/caseyjlaw/sdmreader']
)
