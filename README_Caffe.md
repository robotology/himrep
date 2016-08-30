caffeCoder Installation & Setup
======

## Notes

Although Caffe can be compiled also on the CPU, it is recommended to run this module on an NVIDIA GPU with Compute Capability higher or equal to 3.0 and CUDA version higher or equal to 7.0, in order to obtain high performance at runtime by exploiting NVIDIA cuDNN library v3 or higher (cuDNN v1 and 2 are not supported by Caffe current release, which is the one supported by this module).

At present, the module has been tested on:

- `Tesla K40`: around `10 ms` per image
- `GeForce 650M`: around `50 ms` per image
- `Quadro K2200`: around `60 ms` per image

These numbers are obtained with the simplest use of the provided Caffe's wrapper (`CaffeFeatExtractor` class), i.e., extracting features from one image at a time. Higher performances can be obtained extracting features from batches of images.

## Dependencies

- [OpenCV](http://opencv.org/downloads.html)
This a required dependency of both Caffe and caffeCoder module.
- [Caffe](http://caffe.berkeleyvision.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone)
This is an optional dependency of Caffe as described in Caffe instructions. However at present CUDA is also a required dependency of the module.
Indeed CUDA events are used in the module to measure the feature extraction time on the GPU. Hence, if you need to use this module without CUDA, and with Caffe running on the CPU, you'll have to (i) remove the CUDA dependency from CaffeFeatExtractor.hpp (removing CUDA includes and deleting the timing code) and (ii) remove the CUDA related instructions from the CMakeLists.txt inside caffeCoder/src.

## Caffe Installation

For a complete and continuously updated guide to how to install Caffe in any configuration you should go to [Caffe - Installation](http://caffe.berkeleyvision.org/installation.html).
We do not cover here exhaustively the procedure because it is changing quite rapidly (as Caffe is under active developement) and detailed instructions are provided by the developers. We just report the procedure we followed at present (06/16/16) to use Caffe library in YARP and iCub's modules on Ubuntu 16.04 LTS.

##### CUDA installation

Download and install CUDA drivers and toolkit by following [CUDA Installation Guide for Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4BkDT7m6r). CUDA 7.0 or later is strongly recommended in order to exploit the speedup offered by the [NVIDIA cuDNN library v3 or higher](https://developer.nvidia.com/cuDNN) needed by Caffe current release. 


**Important Notes**:

- If you choose the Runfile Installation, remember to answer **`YES`** if asked about **`DKMS`**.
- We found very useful the instructions available at this [link](http://yohan.jasdid.com/2015/03/installing-nvidia-drivers-and-cuda-7-0-in-debian-wheezy/) to install CUDA 7.0 on a Debian `wheezy` system.
- At present (06/16/16) the only toolkit version officially supporting Ubuntu 16.04 and GCC 5.3 is CUDA 8 Release Candidate. Hence, until the CUDA 8 Production Release will be published, we temporarily installed CUDA 7.5 on Ubuntu 16.04 by:

```
sudo apt-get install nvidia-cuda-dev nvidia-cuda-toolkit
```
However, in order to compile Caffe with this CUDA version on Ubuntu 16.04 and GCC 5.3, you'll need to apply a little workaround (that we describe in the following). Therefore, we recommend to install CUDA 8 when it will be available for Ubuntu 16.04.

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

Where in the CMake configuration you should have set the installation path (CMAKE_INSTALL_PREFIX) to one of your choice. In this case, set the `OpenCV_DIR` environment variable to the installation path to allow Caffe finding it.

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

At present we support the first and only [Caffe release candidate 3](https://github.com/BVLC/caffe/releases) that is version 1.0.0.rc3. You can download it from the linked page.

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

(NOTE *) We point out that, in the configuration step:

- you should be able to link to all installed dependencies, if you have set correctly the environment variables
- set BLAS to `open` or `Open` if you installed OpenBLAS as we did: if you still see that the Atlas implementation is not found, this might be an issue with Caffe CMake: in any case, if you check in the advanced mode, you should see that OpenBLAS has been found in your installation directory
- there is no need to build the Matlab or Python wrappers for Caffe
- the benchmarks reported above have been obtained using the cuDNN library thus it is better to use it if possible (set USE_CUDNN to ON)
- **important if you are on Ubuntu 16.04 and use GCC 5.3 with CUDA 7.5**: as noted [here](https://github.com/BVLC/caffe/issues/4046), in this case you need to modify the `CMAKE_CXX_FLAGS` CMake variable by appending to it the `-D_FORCE_INLINES` flag. You can do it in the `CMakeLists.txt` by modifying the following line:

	```
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_FORCE_INLINES -Wall")
	```
or during the interactive configuration step with `ccmake`. Alternatively, you can use CUDA 8 as suggested above.

Finally, set the `Caffe_DIR` environment variable to the installation path to allow finding Caffe via `find_package`.

## caffeCoder Setup

Some data is necessary to extract features from a learned network model in Caffe. The setup procedure basically follows the [Feature extraction with Caffe C++ code](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html) example provided with Caffe. If you are not interested in the details, you can simply execute the following instructions and use the default network model and parameters. We provide a little more explanataion hereafter for those who want to play with the module's models and settings.

Set the `Caffe_ROOT` environment variable to your Caffe's source root directory.

1. Provide the weights of the network model:<br>

    ```
    cd $Caffe_ROOT && scripts/download_model_binary.py models/bvlc_reference_caffenet
    ```
2. Provide the mean image:<br>

    ```
    cd $Caffe_ROOT && ./data/ilsvrc12/get_ilsvrc_aux.sh
    ```
3. Provide the network model definition:<br>

	- Install the `imagenet_val_cutfc6.prototxt` file located in the `himrep` repository (or the whole himrep context) into the YARP local context directory:<br>
`yarp-config context --import himrep imagenet_val_cutfc6.prototxt`<br>
	- Open the imported file (should be in `~/.local/share/yarp/contexts/himrep/imagenet_val_cutfc6.prototxt`) and modify the absolute path to the mean image (`mean_file` field in the `transform_param` section) with the correct path to the downloaded mean image (see step 2) on your machine, without using environment variables (if you have followed the instructions above, this path should be `$Caffe_ROOT/data/ilsvrc12/imagenet_mean.binaryproto` with the value of `$Caffe_ROOT` on your machine substituted).

Now you are ready to compile and start playing with the module!

##### Detailed explanation

1. Provide the weights of the network model:<br>

	In Caffe framework, these are stored in a .caffemodel file, whose absolute path must be provided to the module in the `pretrained_binary_proto_file` parameter.
In Caffe's [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) there are many models available with related descriptions and usage instructions. If you choose the *BVLC Reference CaffeNet* model, as we did, you can download the weights (and other data related to the model) by running the following command from Caffe's source root directory:<br>

	```
	scripts/download_model_binary.py models/bvlc_reference_caffenet
	```
	

	This creates the file `models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel`. If you set the `Caffe_ROOT` environment variable to Caffe's source root directory, in case you do not provide any `pretrained_binary_proto_file` parameter to the module, this is set by default to `$Caffe_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel`; otherwise, if such variable does not exist, in case you do not provide the parameter this is set to empty string and the module fails to start.

2. Provide the mean image and network model definition:<br>

	This is represented in Caffe framework by a .prototxt file, whose name must be provided to the module in the `feature_extraction_proto_file` parameter and whose location must be in the Resouce Finder search path.<br>
This file usually is a modified copy of the .prototxt file that comes with the downloaded model (see above). In particular:
  - If the purpose is feature extraction, all layers after the one from which one wants to extract the output can be deleted to avoid unnecessary computations. That's why the default value of this parameter is the file `imagenet_val_cutfc6.prototxt`, because the layer we extract the output from is `fc6` by default (see `extract_feature_blobs_name` parameter).
  - The input (data) layer does not depend on the model and in general it can be changed depending on how one wants to provide the images to the network (see [Data Layers Catalogue](http://caffe.berkeleyvision.org/tutorial/layers.html#data-layers)). In this module we use a *Memory Data Layer* therefore you will find it in the provided .prototxt files. The only parameter that you need to modify in any case is the path to the mean image that is subtracted from each input image before feeding it to the network. This is the mean image of the training set on which the model has been learned. For the chosen model, you can download it by running this script from Caffe's source root directory: 

		```
		./data/ilsvrc12/get_ilsvrc_aux.sh
		```
This creates the file `data/ilsvrc12/imagenet_mean.binaryproto` and you must set its absolute path (without using environment variables) in the file `imagenet_val_cutfc6.prototxt`.

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
