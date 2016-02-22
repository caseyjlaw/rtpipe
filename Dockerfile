# Builds rtpipe by starting with anaconda to build pwkit
FROM continuumio/anaconda:latest

RUN apt-get install -y debian-archive-keyring
RUN apt-key update -y

RUN apt-get update && apt-get install -y libfftw3-bin libfftw3-dev python-pip
RUN conda install -y -c pkgw casa-data casa-python 
RUN pip install rtpipe sdmreader sdmpy pwkit pyfftw
