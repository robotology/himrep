caffeCoder Setup
======

## Notes

Although Caffe can be compiled also on the CPU, it is recommended to run this module on an NVIDIA GPU with Compute Capability higher or equal to 3.0 and CUDA version higher or equal to 5.5 in order to obtain acceptable performance at runtime.

At present, the module has been tested on:
    - Tesla K40: around 10-13ms per image
    - GeForce 650M: around 45-50ms per image
These numbers are obtained with the simplest use of the provided Caffe's wrapper (CaffeFeatExtractor class), i.e., extracting features from one image at a time. Higher performances can be obtained extracting features from mini-batches of images.

## Dependencies

- [OpenCV](http://opencv.org/downloads.html)
This a required dependency of both Caffe and caffeCoder module.
- [Caffe](http://caffe.berkeleyvision.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone)
This is an optional dependency of Caffe as described in Caffe instructions. However at present CUDA is also a required dependency of the module.
Indeed CUDA events are used for timing the feature extraction on the GPU. Still, you can use this module running Caffe on the CPU if you do not have CUDA installed, because this is the only direct dependency of the module on CUDA at the moment. You'll have to (i) remove the CUDA dependency from CaffeFeatExtractor.hpp (removing CUDA includes and deleting the timing code) and (ii) remove the CUDA related instructions from the CMakeLists.txt inside caffeCoder/src.

## Caffe Installation

For a complete guide to how to install Caffe in any configuration you can go to [Caffe - Installation](http://caffe.berkeleyvision.org/installation.html).
We do not cover here exhaustively the procedure because it is changing quite rapidly (as Caffe is under active developement) and updated and detailed instructions are provided by the developers. We report the procedure we followed to use Caffe library in YARP and iCub's modules on Ubuntu 14.04 LTS.

##### CUDA installation

Download and install CUDA drivers and toolkit following [CUDA Getting started guide for Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#axzz3VV5Adq1a). CUDA 6.5 or later is strongly recommended in order to exploit the speedup offered by the [NVIDIA cuDNN library](https://developer.nvidia.com/cuDNN). We installed CUDA 6.5 but now also CUDA 7 is available and supported by Caffe. If the desired CUDA package isn't available yet with the package management system of Ubuntu (CUDA 6.5 is not, at the time being) we recommend installing the NVIDIA proprietary .run instead of adding external repositories, in order to have a stable system.

##### cuDNN installation (optional but recommended)

Download **cuDNN** from [NVIDIA cuDNN library](https://developer.nvidia.com/cuDNN) (you have to sign up as CUDA Registered Developer, it's for free), extract the archive and copy the libraries and the headers inside the CUDA directories:

```
tar -xzvf cudnn-6.5-linux-R1.tgz
cd cudnn-6.5-linux-R1
sudo cp lib* /usr/local/cuda/lib64/
sudo cp cudnn.h /usr/local/cuda/include/
```

We installed the cuDNN Release 1 but now also the Release 2 is available and should work fine with Caffe.

##### BLAS installation

We chose the **OpenBLAS** implementation but also ATLAS or Intel MKL are supported by Caffe.
Download the source code from [OpenBLAS page](http://www.openblas.net/) and follow instructions to compile and install it. We installed the version `0.2.13`. We recommend to install OpenBLAS from source in a separate and specified location of your choice (instead of the default `/usr/local`, where e.g. the `libopenblas-base` package that comes with Ubuntu 14.04 is installed) because in this case there might be some linking issues with some YARP libraries (e.g. YARP_math).
Commonly, it is sufficient to do:

```
tar -xzvf name.tar.gz
cd name
make PREFIX=/path/to/install/dir install
```

Set the `OpenBLAS_HOME` environment variable to the installation path to allow Caffe finding it.

##### BOOST installation

Download the source code from [Boost C++ Libraries](http://www.boost.org/) and follow instructions to compile and install it. We installed the version 1.55. For convenience, we report the followed instructions (that can be found on Boost page):

```
tar --bzip2 -xf /path/to/boost_1_55_0.tar.bz2
./bootstrap.sh --prefix=path/to/install/dir
./b2 install
```

Set the `Boost_DIR` environment variable to the installation path to allow Caffe finding it.

##### OpenCV installation

Download the source code from [OpenCV - Downloads](http://opencv.org/downloads.html). We installed the version 2.4.10; OpenCV 3.0 is not supported yet by YARP and iCub software. You can do:

```
unzip OpenCV-$version.zip
cd opencv-$version
mkdir build
cd build
ccmake ../
make
make install
```

Where in the CMake configuration you might have set the installation path (CMAKE_INSTALL_PREFIX) to one of your choice.

Set the `OpenCV_DIR` environment variable to the installation path to allow Caffe finding it.

##### Other packages

Refer to [Caffe - Ubuntu Installation](http://caffe.berkeleyvision.org/install_apt.html) for updated instructions. On Ubunutu 14.04 LTS at the time being we have done:

Google Protobuf Buffers C++:<br>
`sudo apt-get install libprotobuf-dev protobuf-compiler`

Google Logging:<br>
`sudo apt-get install libgoogle-glog-dev`

Google Flags:<br>
`sudo apt-get install libgflags-dev`

LevelDB:<br>
`sudo apt-get install libleveldb-dev`

HDF5:<br>
`sudo apt-get install libhdf5-serial-dev`

LMDB:<br>
`sudo apt-get install liblmdb-dev`

snappy:<br>
`sudo apt-get install libsnappy-dev`

##### Caffe compilation

Clone the [Caffe Github repository](https://github.com/BVLC/caffe):<br>
`git clone https://github.com/BVLC/caffe.git`

In order to be able to link Caffe from an external project via CMake (as this application does) you should compile Caffe via CMake and not manually editing the Makefile.config. At present related instructions refer to the PR 1667 [Improved CMake scripts](https://github.com/BVLC/caffe/pull/1667), even if the PR has been merged in Caffe's master branch. Generally you can do:

```
cd caffe
mkdir build
cd build
ccmake ../ (NOTE *)
make all
make runtest
make install
```

(NOTE *) We point out that, in the configuration step:
- you should be able to link to all installed libraries if you have set correctly the environment variables
- set BLAS to `open` or `Open` if you installed OpenBLAS as we did: do not be worried if you still see that the Atlas implementation is not found, this is an issue with Caffe CMake compilation at the moment, but actually if you check you can see that OpenBLAS has been found in your installation directory
- there is no need to build the Matlab or Python wrappers for Caffe
- the benchmarks reported above have been obtained using the cuDNN library thus it is better to use it if possible (set USE_CUDNN to ON)

Set the `Caffe_DIR` environment variable to the installation path to allow finding Caffe via `find_package`.

## caffeCoder Setup

Some data is necessary to extract features from a learned network model in Caffe. The setup procedure basically follows the [Feature extraction with Caffe C++ code](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html) example provided with Caffe. If you are not interested in the details, you can simply execute the following instructions and use the default network model and parameters. We provide a little more explanataion hereafter for those who want to play with the module's models and settings.

Set the `Caffe_ROOT` environment variable to your Caffe's source root directory.

1. Provide the weights of the network model:
```
cd $Caffe_ROOT
scripts/download_model_binary.py models/bvlc_reference_caffenet
```
2. Provide the mean image:
```
cd $Caffe_ROOT
./data/ilsvrc12/get_ilsvrc_aux.sh
```
3. Provide the network model definition:<br>
Install the `imagenet_val_cutfc6.prototxt` file located in the `himrep` repository (or all himrep context) into the YARP local context directory:<br>
`yarp-config context --import himrep imagenet_val_cutfc6.prototxt`<br>
Open the imported file (should be in `~.local/share/yarp/contexts/himrep/imagenet_val_cutfc6.prototxt`) and modify the absolute path to the mean image (`mean_file` field in the `transform_param` section) with the correct path to the downloaded mean image (see step 2) on your machine, without using environment variables (if you have followed the instructions above, this path should be `$Caffe_ROOT/data/ilsvrc12/imagenet_mean.binaryproto` with the value of `$Caffe_ROOT` on your machine substituted).

Now you are ready to compile and start playing with the module!

##### Detailed explanation

1. Provide the weights of the network model.<br>
In Caffe framework, these are stored in a .caffemodel file, whose absolute path must be provided to the module in the `pretrained_binary_proto_file` parameter.
In Caffe's [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) there are many models available with related descriptions and usage instructions. If you choose the *BVLC Reference CaffeNet* model, as we did, you can download the weights (and other data related to the model) by running the following command from Caffe's source root directory:<br>
`scripts/download_model_binary.py models/bvlc_reference_caffenet`<br>
This creates the file `models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel`. If you set the `Caffe_ROOT` environment variable to Caffe's source root directory, in case you do not provide any `pretrained_binary_proto_file` parameter to the module, this is set by default to `$Caffe_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel`; otherwise, if such variable does not exist, in case you do not provide the parameter this is set to empty string and the module fails to start.

2. Provide the mean image and network model definition.<br>
This is represented in Caffe framework by a .prototxt file, whose name must be provided to the module in the `feature_extraction_proto_file` parameter and whose location must be in the Resouce Finder search path.<br>
This file usually is a modified copy of the .prototxt file that comes with the downloaded model (see above). In particular:
  - If the purpose is feature extraction, all layers after the one from which one wants to extract the output can be deleted to avoid unnecessary computations. That's why the default value of this parameter is the file `imagenet_val_cutfc6.prototxt`, because the layer we extract the output from is `fc6` by default (see `extract_feature_blobs_name` parameter).
  - The input (data) layer does not depend on the model and in general it can be changed depending on how one wants to provide the images to the network (see [Data Layers Catalogue](http://caffe.berkeleyvision.org/tutorial/layers.html#data-layers)). In this module we use a *Memory Data Layer* therefore you will find it in the provided .prototxt files. The only parameter that you need to modify in any case is the path to the mean image that is subtracted from each input image before feeding it to the network. This is the mean image of the training set on which the model has been learned. For the chosen model, you can download it by running this script from Caffe's source root directory: `./data/ilsvrc12/get_ilsvrc_aux.sh`. This creates the file `data/ilsvrc12/imagenet_mean.binaryproto` and you must set its absolute path (without using environment variables) in the file `imagenet_val_cutfc6.prototxt`.

## License

Material included here is Copyright of _iCub Facility - Istituto Italiano di Tecnologia_ and is released under the terms of the GPL v2.0 or later. See the file LICENSE for details.
