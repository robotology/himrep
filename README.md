Hierarchical IMage REPresentation
======

## Installation

##### Dependencies

- [YARP](https://github.com/robotology/yarp)
- [iCub](https://github.com/robotology/icub-main)
- [icub-contrib-common](https://github.com/robotology/icub-contrib-common)
- [OpenCV](http://opencv.org/downloads.html)
- [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) ([`linearClassifierModule`](https://github.com/robotology/himrep/tree/master/modules/linearClassifierModule))
- [SiftGPU](http://cs.unc.edu/~ccwu/siftgpu) ([`sparseCoder`](https://github.com/robotology/himrep/tree/master/modules/sparseCoder))
- [Caffe](http://caffe.berkeleyvision.org/) ([`caffeCoder`](https://github.com/robotology/himrep/tree/master/modules/caffeCoder))
- [CUDA](https://developer.nvidia.com/cuda-zone) ([`caffeCoder`](https://github.com/robotology/himrep/tree/master/modules/caffeCoder))

The `liblinear-1.91` package is a dependency for the `linearClassifierModule` and should be thus compiled beforehand. The resulting static library must be placed in the same directory of the `liblinear` package to then provide the environment variable **`LIBSVMLIN_DIR`** pointing to that location.

The `SiftGPU` library is a dependency for the `sparseCoder` module.

The `Caffe` library and `CUDA` package are dependencies for the `caffeCoder` module. Documentation about `caffeCoder` setup can be found in the [README_Caffe](https://github.com/robotology/himrep/blob/master/README_Caffe.md).

## Documentation

This repository contains a collection of modules that perform features extraction and learning through a linear classifier.

Online documentation is available here: [http://robotology.github.com/himrep](http://robotology.github.com/himrep).

## License

Material included here is Copyright of _iCub Facility - Istituto Italiano di Tecnologia_ and is released under the terms of the GPL v2.0 or later. See the file LICENSE for details.
