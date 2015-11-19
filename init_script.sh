#!/bin/sh

module add matlab-8.6
module add python-2.7.6-gcc
module add python27-modules-gcc
module add atlas-3.10.1-gcc4.7.0
module add opencv-2.4
module add ffmpeg
module add hdf5-1.8.12-gcc-serial

export PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/gogartom/DOM-Extraction/caffe-fast-rcnn/build/install/python
export PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/gogartom/easydict/lib/python2.7/site-packages/
export PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/gogartom/scikit-image/lib/python2.7/site-packages/
export PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/gogartom/protobuf_py_2.6/lib/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/gogartom/
export PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/gogartom/DOM-Extraction/lib/

export PATH=$PATH:/afs/ics.muni.cz/software/hdf5-1.8.12/gcc-serial
export PATH=$PATH:/storage/plzen1/home/gogartom/DOM-Extraction/caffe-fast-rcnn/build/install/bin

