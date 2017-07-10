Table of Contents
=================
  * [Description](#description)
  * [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Compilation](#compilation)
  * [Setup](#setup)
    * [Download binary Caffe models (.caffemodel)](#download-binary-caffe-models-caffemodel)
    * [Configure .ini](#configure-ini)
  * [Detailed explanation](#detailed-explanation)
  * [License](#license)

## Description

This module is a basic YARP wrapper for [TensorRT](https://developer.nvidia.com/tensorrt), which receives as input a stream of images (of type `yarp::sig::Image`), feeds them to a Convolutional Neural Network (CNN) model and produces as output a corresponding stream of vectors (of type `yarp::sig::Vector`), which can be extracted from any CNN layer.

## Installation

### Dependencies

The libraries that are needed to compile this module are:

- [YARP](https://github.com/robotology/yarp) 
- [iCub](https://github.com/robotology/icub-main) 
- [OpenCV](http://opencv.org/releases.html) 
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn): these are dependencies of TensorRT and also of the module itself

### Compilation

Provided that the dependencies are satified, you can compile this module just by setting the `BUILD_GIECoder` flag to `ON` as explained [here](https://www.github.com/robotology/himrep/README/#compilation).

When you run the `ccmake` command, ensure also that:

- `TensorRT` is correctly found on the system
- the flag `CUDA_USE_STATIC_CUDA_RUNTIME` is set to `OFF`

## Setup

This module can use arbitrary Caffe models. In the following, we report istructions on how to setup the module to use either one of two well-known models, `ResNet-50` and `CaffeNet`.

If you are not interested in the details, you can execute the following instructions and use one of these two models with the default module parameters. We provide a little more explanataion hereafter for those who want to try different or custom Caffe models and settings.

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

by downloading, in this folder, the `ResNet-50-model.caffemodel` and the `ResNet-50-deploy.prototxt` files from the [OneDrive link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777) specified at the webpage.

### Configure .ini

We have now to customize the module's `.ini` file in order to use the downloaded Caffe model. Some `.ini` examples are provided (e.g. for the two networks considered here plus `GoogLeNet`) with the source code of the module (inside `app/conf`). Therefore we can import such `.ini` files from `himrep` for the `CaffeNet` and `ResNet-50` by doing:

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
prototxt_file $Caffe_ROOT/models/bvlc_reference_caffenet/deploy.prototxt
~~~

replacing the `$Caffe_ROOT` env variable with its full value.

For `ResNet-50` do:

~~~
gedit caffeCoder_resnet.ini
~~~

And then set:

~~~
caffemodel_file $Caffe_ROOT/models/ResNet-50/ResNet-50-model.caffemodel
prototxt_file $Caffe_ROOT/models/ResNet-50/ResNet-50-deploy.prototxt
~~~

replacing the `$Caffe_ROOT` env variable with its full value.

For other parameters and input and output ports we refer to the module documentation [here](http://robotology.github.io/himrep/doxygen/doc/html/group__GIECoder.html).

## Detailed explanation

In Caffe, the weights of network models are stored in a `.caffemodel` file, whose absolute path must be provided to the `GIECoder` in the `caffemodel_file` parameter. The network definition file to be used in inference mode instead is usually the `deploy.prototxt` and its absolute path must be provided to the `GIECoder` in the `prototxt_file` parameter.

In Caffe's [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) there are many models available with related descriptions and usage instructions. You can use them with `GIECoder`, just by passing the path to their `deploy.prototxt` (and ensuring that TensorRT can convert them correctly, by checking, e.g., that there are no layer kinds which are not supported by the engine).

In order to correctly use a network in inference mode, the mean image (or pixel) of the training set that has been used to learn the model parameters must be subtracted from any image that is fed to the model. The mean image is usully stored in Caffe with a `.binaryproto` file. You will need to specify this information in the `.ini`:

- if the mean image is subtracted, the `.binaryproto` file must be pointed by the `binaryproto_meanfile` parameter;
- if the mean pixel is subtracted, you will need to specify, in the `.ini` file, five additional parameters: three of them are the R, G, B, values of the pixel (`meanR`, `meanG`, `meanB`) and two of them are the width and height to which the input image will be resized before being fed to the network (`resizeWidth` and `resizeHeight`).

Another important parameter to be set in the `.ini` file is the tag/name of the output of the layer we want to read. This can be specified by setting the `blob_name` parameter.

## License

Material included here is Copyright of _iCub Facility - Istituto Italiano di Tecnologia_ and is released under the terms of the GPL v2.0 or later. See the file LICENSE for details.
