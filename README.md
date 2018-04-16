Hierarchical IMage REPresentation
=================================

<a href="https://zenhub.com"><img src="https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png"></a>

Table of Contents
=================

  * [Repository Structure](#repository_structure)
  * [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Compilation](#compilation)
  * [Documentation](#documentation)
  * [Contributing](#license)

## Repository Structure

This repository contains a collection of modules to extract features from images or to perform classification tasks on feature vectors. These modules are meant to be used by other demos to build object recognition pipelines.

At present, the following modules for feature extraction are available:

- `caffeCoder`
- `GIECoder`
- `sparseCoder`

Each of them takes as input an image and outputs its vector representation.

The `linearClassifierModule` instead implements a linear classifier which can be trained and tested on feature vectors. It is included in this repository because its main usage so far has been on top of a feature extraction module in order to perform image classification, but it can be used on any kind of vectors. While the module is currently in use on our platforms providing good performance, we are working to upgrade it in order to make it faster and more accurate.

## Installation

### Dependencies

While

- [YARP](https://github.com/robotology/yarp)
- [iCub](https://github.com/robotology/icub-main)
- [icub-contrib-common](https://github.com/robotology/icub-contrib-common)
- [OpenCV](http://opencv.org/downloads.html) (version < 3.0 is required by `sparseCoder`)

are needed by all modules, the following dependencies are required only if you plan to build the corresponding module:

- [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/): needed by `linearClassifierModule`
- [SiftGPU](https://github.com/pitzer/SiftGPU): needed by `sparseCoder`
- [Caffe](http://caffe.berkeleyvision.org/): needed by `caffeCoder`
- [TensorRT](https://developer.nvidia.com/tensorrt): needed by `GIECoder`
- [CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn): optional for `caffeCoder` but mandatory for `GIECoder`

Instructions on how to setup the dependencies for each module can be found in specific README files:

- `caffeCoder`: link to [README](modules/caffeCoder)
- `GIECoder`: link to [README](modules/GIECoder)
- `sparseCoder`: link to [README](modules/sparseCoder)
- `linearClassifierModule`: link to [README](modules/linearClassifierModule)

### Compilation

Get the code:

~~~
$ git clone https://github.com/robotology/himrep.git
~~~

And then do, as usual:

~~~
$ cd himrep
$ mkdir build && cd build
$ ccmake ../
~~~

Where you will configure the project by setting to `ON` the modules you want to compile and to `OFF` the ones you want to skip.

**IMPORTANT** When you run the `ccmake` command, ensure that:

- the `CMAKE_INSTALL_PREFIX` points to the `icub-contrib-common` installation directory
- the `YARP_DIR`, `ICUB_DIR`, `OpenCV_DIR` are correctly pointing to valid installation paths

After that, you can compile and install as usual:
~~~
$ make
$ make install
~~~

## Documentation

Online autogenerated documentation is available here: [http://robotology.github.com/himrep](http://robotology.github.com/himrep).

## License

Material included here is Copyright of _iCub Facility - Istituto Italiano di Tecnologia_ and is released under the terms of the GPL v2.0 or later. See the file LICENSE for details.
