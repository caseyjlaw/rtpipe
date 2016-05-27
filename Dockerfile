# Builds rtpipe by starting with anaconda to build pwkit
FROM continuumio/anaconda:latest

RUN apt-get install -y debian-archive-keyring
RUN apt-key update -y

RUN apt-get update && apt-get install -y libfftw3-bin libfftw3-dev python-pip
RUN conda install -y -c pkgw numpy scipy casa-data casa-python casa-tools jupyter bokeh cython matplotlib
RUN pip install rtpipe
