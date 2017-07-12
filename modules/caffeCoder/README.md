Table of Contents
=================
  * [Description](#description)
  * [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Compilation](#compilation)
  * [Setup](#setup)
    * [Download binary Caffe models (.caffemodel)](#download-binary-caffe-models-caffemodel)
    * [Configure .prototxt and .ini with absolute paths](#configure-prototxt-and-ini-with-absolute-paths)
  * [Detailed explanation](#detailed-explanation)
  * [Additional notes on Caffe installation](#additional-notes-on-caffe-installation)
  * [Citation](#citation)
  * [License](#license)

## Description

This module is a basic YARP wrapper for [Caffe](http://caffe.berkeleyvision.org/), which receives as input a stream of images (of type `yarp::sig::Image`), feeds them to a Convolutional Neural Network (CNN) model and produces as output a corresponding stream of vectors (of type `yarp::sig::Vector`), which can be extracted from any CNN layer.

## Installation

### Dependencies

The libraries that are needed to compile this module are:

- [YARP](https://github.com/robotology/yarp)
- [iCub](https://github.com/robotology/icub-main)
- [OpenCV](http://opencv.org/releases.html)
- [Caffe](https://www.github.com/BVLC/caffe.git)
- [CUDA](https://developer.nvidia.com/cuda-zone):
this is an optional dependency of Caffe and also of this module. However, we strongly recommend to rely on a powerful enough NVIDIA CUDA-enabled GPU (with Compute Capability >= 3.0) in order to achieve a good frame rate.

### Compilation

Provided that the dependencies are satified, you can compile this module just by setting the `BUILD_caffeCoder` flag to `ON` as explained [here](https://www.github.com/robotology/himrep#compilation).

When you run the `ccmake` command, ensure also that:

- the `Caffe_DIR` flag is correctly pointing to a valid Caffe installation
- the flag `CUDA_USE_STATIC_CUDA_RUNTIME` is set to `OFF`

## Setup

This module can use arbitrary Caffe models: the only contraint is that the model must have an input data layer of kind [MemoryData](http://caffe.berkeleyvision.org/tutorial/layers/memorydata.html). However, this can be easily achieved for most models just by replacing the input data layer in the model definition file. In the following, we report istructions on how to setup the module to use either one of two well-known models, `ResNet-50` and `CaffeNet`.

The setup procedure basically follows the [Extracting Features](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html) example provided with Caffe. If you are not interested in the details, you can execute the following instructions and use one of these two models with the default module parameters. We provide a little more explanataion hereafter for those who want to try different or custom Caffe models and settings.

### Download binary Caffe models (.caffemodel)

If you don't have setted it already, for convenience, set the `Caffe_ROOT` env variable pointing to your `caffe` source code directory.

For `CaffeNet` we can follow the instructions on [caffe](http://caffe.berkeleyvision.org/model_zoo.html) website:

~~~
$ cd $Caffe_ROOT
$ scripts/download_model_binary.py models/bvlc_reference_caffenet
# for this model we need also to get the mean image of the training set of ILSVRC
$ ./data/ilsvrc12/get_ilsvrc_aux.sh
~~~

For `ResNet-50` we can get the weights as indicated [here](https://github.com/KaimingHe/deep-residual-networks):

~~~
$ cd $Caffe_ROOT/models
$ mkdir ResNet50
$ cd ResNet50
~~~

by downloading, in this folder, the `ResNet-50-model.caffemodel` file from the [OneDrive link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777) specified at the webpage.

### Configure .prototxt and .ini with absolute paths

We have now to customize two files in order to use the downloaded Caffe model.
These are provided for some Caffe models (the two considered here plus `GoogLeNet`) with the source code of the module (inside `app/conf`) and are:

- a `.ini` file for the `caffeCoder` module, specifying some parameters and pointing to the Caffe model's binary and definition files
- the `.prototxt` Caffe model's definition file, where we have replaced the input data layer with one of kind `MemoryData`

#### File .prototxt: configuration

Import the `.prototxt` files from `himrep` for the `CaffeNet` and `ResNet-50` architectures in order to customize them for your system:

~~~
$ yarp-config context --import himrep bvlc_reference_caffenet_val.prototxt
$ yarp-config context --import himrep resnet_val.prototxt
~~~

For `CaffeNet` we need to do:

~~~
$ cd ~/.local/share/yarp/contexts/himrep
$ gedit bvlc_reference_caffenet_val.prototxt
~~~

At line 10, replace the following:

~~~
mean_file: "/path/to/train_mean.binaryproto"
~~~

with the correct absolute path to this file on your system (without using env variables). This is the binary file containing the mean image of the training set of ILSVRC we just downloaded. In this case this is:

~~~
mean_file: "$Caffe_ROOT/data/ilsvrc12/imagenet_mean.binaryproto"
~~~

where you must replace the `$Caffe_ROOT` env variable with its full value.

For `ResNet-50` we do not need to modify anything.

#### File .ini: configuration

Import the `.ini` files from `himrep` for the `CaffeNet` and `ResNet-50` in order to customize them on your system:

~~~
# CaffeNet
$ yarp-config context --import himrep caffeCoder_caffenet.ini
$ yarp-config context --import himrep caffeCoder_resnet.ini
~~~

For each of them we must set the following variables to the correct paths: `caffemodel_file` and `prototxt_file`.

For `CaffeNet` do:

~~~
$ cd ~/.local/share/yarp/contexts/himrep
$ gedit caffeCoder_caffenet.ini
~~~

And then set:

~~~
caffemodel_file $Caffe_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
prototxt_file ~/.local/share/yarp/contexts/himrep/bvlc_reference_caffenet_val.prototxt
~~~

replacing the `$Caffe_ROOT` env variable and `~/` with their full values.

For `ResNet-50` do:

~~~
gedit caffeCoder_resnet.ini
~~~

And then set:

~~~
caffemodel_file $Caffe_ROOT/models/ResNet50/ResNet-50-model.caffemodel
prototxt_file ~/.local/share/yarp/contexts/himrep/resnet_val.prototxt
~~~

replacing the `$Caffe_ROOT` env variable and `~/` with their full values.

For other parameters and input and output ports we refer to the module documentation [here](http://robotology.github.io/himrep/doxygen/doc/html/group__caffeCoder.html).

## Detailed explanation

In Caffe, the weights of network models are stored in a `.caffemodel` file, whose absolute path must be provided to the `caffeCoder` in the `caffemodel_file` parameter. The network definition file to be used in inference mode instead is usually called `deploy.prototxt` and must be provided to the `caffeCoder` in the `prototxt_file` parameter.

In Caffe's [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) there are many models available with related descriptions and usage instructions. You can use them with `caffeCoder`, by copying their `deploy.prototxt` and replacing their input layer with a corresponding `MemoryData` layer.

In order to correctly use a network in inference mode, the mean image (or pixel) of the training set that has been used to learn the model parameters must be subtracted from any image that is fed to the model. The mean image is usully stored in Caffe with a `.binaryproto` file. You will to specify this in the `MemoryData` layer and:

- if the mean image is subtracted, the `.binaryproto` file must be downloaded and correctly pointed by the `.prototxt` in your system (`mean_file` field of the `MemoryData` layer);
- if the mean pixel is subtracted, you will need to specify, in the `.ini` file, two additional parameters (as, e.g., we do in `caffeCoder_googlenet.ini`) related to the width and height (`resizeWidth` and `resizeHeight`) to which the input image will be resized before being fed to the `MemoryData` layer.

Another important parameter to be set in the `.ini` file is the tag/name of the output of the layer we want to read. This can be specified by setting the `blob_name` parameter.

## Additional notes on Caffe installation

For a complete and continuously updated guide to how to install Caffe in any configuration you should go to [Caffe - Installation](http://caffe.berkeleyvision.org/installation.html).
We do not cover here exhaustively the procedure. We just report the procedure we followed at present (07/2017) to use Caffe from this module on Ubuntu 16.04 LTS.

##### CUDA installation

Download and install CUDA drivers and toolkit by following [CUDA Installation Guide for Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4BkDT7m6r).

##### cuDNN installation (optional but recommended)

Download the **cuDNN** version you need (depending on the toolkit version) from [NVIDIA cuDNN library](https://developer.nvidia.com/cuDNN) (you have to sign up as CUDA Registered Developer, it's for free), and install it by following the instructions.

##### BLAS installation

We chose the **OpenBLAS** implementation but also ATLAS or Intel MKL are supported by Caffe.
You can either download the source code from [OpenBLAS page](http://www.openblas.net/) and follow instructions to compile and install it, or install the package.
In the latter case, you can just do:

```
sudo apt-get install libopenblas-dev
```

In case you compile the source code, we recommend to install in a separate and specified location of your choice instead of the default `/usr/local` by doing:

```
tar -xzvf <downloaded-openblas-archive-name>.tar.gz
cd <downloaded-openblas-archive-name>
make PREFIX=/path/to/install/dir install
```
and setting the `OpenBLAS_HOME` environment variable to the installation path to allow Caffe finding it.

##### BOOST installation

As for BLAS, you can either download the source code from [Boost C++ Libraries](http://www.boost.org/) and follow instructions to compile and install it, or install the package. In any case, check the supported versions on Caffe website.
For convenience, again we report the followed instructions (that can be found on Boost page) to compile from source:

```
tar --bzip2 -xf <downloaded-boost-archive>.tar.bz2
cd <downloaded-boost-archive>
./bootstrap.sh --prefix=path/to/install/dir
./b2 install
```
and set the `Boost_DIR` environment variable to the installation path to allow Caffe finding it, or to download the package:

```
sudo apt-get install libboost-all-dev
```

##### OpenCV installation

OpenCV comes with the `icub-common` package. In case you need to compile it from source, you can download the source code from [OpenCV - Downloads](http://opencv.org/downloads.html) and compile it through the usual:

```
unzip <downloaded-opencv-archive-name>.zip
cd <downloaded-opencv-archive-name>
mkdir build && cd build
ccmake ../
make
make install
```

Where in the CMake configuration you should have set the installation path (`CMAKE\_INSTALL\_PREFIX`) to one of your choice. In this case, set the `OpenCV_DIR` environment variable to the installation path to allow Caffe finding it.

##### Other packages

Refer to [Caffe - Ubuntu Installation](http://caffe.berkeleyvision.org/install_apt.html) for updated instructions or manual installation. On Ubunutu 16.04 LTS at the time being we have done:

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

Since Caffe is under active development, we try to be compatible with the changes progressively introduced in the framework and periodically check the compatibility of `caffeCoder` against its `master` branch. Therefore you can refer to it and clone it:

```
git clone https://www.github.com/BVLC/caffe.git
```

Note that at present [Caffe RC3](https://github.com/BVLC/caffe/releases) is not compatible anymore with `caffeCoder`.

In order to be able to link Caffe from an external project via CMake (as this application does) you should compile Caffe via CMake and not manually editing the Makefile.config.

Related instructions can be found at [Caffe - Installation](http://caffe.berkeleyvision.org/installation.html) or [here](https://github.com/BVLC/caffe/pull/1667). Generally you can do:

```
cd caffe
mkdir build
cd build
ccmake ../ (NOTE *)
make all
make runtest
make install
```

**NOTE** In the configuration step:

- you should be able to link to all installed dependencies, if you have set correctly the environment variables
- set BLAS to `open` or `Open` if you installed OpenBLAS as we did: if you still see that the Atlas implementation is not found, this might be an issue with Caffe: in any case, if you check by toggling the advanced mode, you should see that OpenBLAS has been found in your installation directory
- there is no need to build the Matlab wrapper for Caffe
- use the cuDNN library if possible (set USE_CUDNN to ON)
- **important if you are on Ubuntu 16.04 and use GCC 5.3 with CUDA 7.5**: as noted [here](https://github.com/BVLC/caffe/issues/4046), in this case you need to modify the `CMAKE_CXX_FLAGS` CMake variable by appending to it the `-D_FORCE_INLINES` flag. You can do it during the interactive configuration step with `ccmake` or by modifying the following line in the `CMakeLists.txt`:

	```
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_FORCE_INLINES -Wall")
	```

Finally, set the `Caffe_DIR` environment variable to the installation path to allow finding Caffe via `find_package`.

## Citation

This module has been presented and benchmarked in the iCub scenario in the following paper:

[Teaching iCub to recognize objects using deep Convolutional Neural Networks](http://jmlr.csail.mit.edu/proceedings/papers/v43/pasquale15.pdf) *Giulia Pasquale, Carlo Ciliberto, Francesca Odone, Lorenzo Rosasco and Lorenzo Natale*,
Proceedings of The 4th Workshop on Machine Learning for Interactive Systems, pp. 21–25, 2015

    @inproceedings{pasquale15,
  	author  = {Giulia Pasquale and Carlo Ciliberto and Francesca Odone and Lorenzo Rosasco and Lorenzo Natale},
  	title   = {Teaching iCub to recognize objects using deep Convolutional Neural Networks},
  	journal = {Proceedings of the 4th Workshop on Machine Learning for Interactive Systems, 32nd International Conference on Machine Learning},
  	year    = {2015},
  	volume  = {43},
  	pages   = {21--25},
  	url     = {http://www.jmlr.org/proceedings/papers/v43/pasquale15}
	}

## License

Material included here is Copyright of _iCub Facility - Istituto Italiano di Tecnologia_ and is released under the terms of the GPL v2.0 or later. See the file LICENSE for details.
